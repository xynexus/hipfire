//! Authoritative bind reporting.
//!
//! `hipfire status` used to derive the server address from the *caller's*
//! config, which silently lies whenever the running process was started with
//! different flags or an older config. The listener's own `local_addr()` is the
//! only authoritative answer, so `serve` records it here and `/health` reports
//! it.

use std::net::{IpAddr, SocketAddr};

use serde_json::{json, Value};

/// What the server is actually listening on, captured from the bound listener.
#[derive(Debug, Clone)]
pub struct BindInfo {
    /// The address the listener reports, after the OS resolved port 0 and any
    /// wildcard host.
    pub addr: SocketAddr,
    /// Concrete addresses a client can reach. For a wildcard bind this is the
    /// enumerated interface list; otherwise it is just `addr`.
    pub addresses: Vec<SocketAddr>,
}

impl BindInfo {
    pub fn capture(addr: SocketAddr) -> Self {
        let addresses = if addr.ip().is_unspecified() {
            let mut addrs = local_interface_addrs()
                .into_iter()
                // A v4 wildcard cannot be reached over v6 and vice versa; only
                // list addresses in the family that is actually bound.
                .filter(|ip| ip.is_ipv4() == addr.is_ipv4())
                .map(|ip| SocketAddr::new(ip, addr.port()))
                .collect::<Vec<_>>();
            addrs.sort_by_key(|a| (a.ip().is_loopback(), a.to_string()));
            addrs.dedup();
            addrs
        } else {
            vec![addr]
        };
        Self { addr, addresses }
    }

    pub fn is_wildcard(&self) -> bool {
        self.addr.ip().is_unspecified()
    }

    pub fn to_json(&self) -> Value {
        json!({
            "addr": self.addr.to_string(),
            "host": self.addr.ip().to_string(),
            "port": self.addr.port(),
            "wildcard": self.is_wildcard(),
            "addresses": self
                .addresses
                .iter()
                .map(|addr| addr.to_string())
                .collect::<Vec<_>>(),
        })
    }
}

/// Enumerate this host's interface addresses via `getifaddrs(3)`.
///
/// Returns an empty vec on any failure: a missing interface list degrades the
/// `/health` listing, and must never fail a health check.
fn local_interface_addrs() -> Vec<IpAddr> {
    let mut out = Vec::new();
    let mut head: *mut libc::ifaddrs = std::ptr::null_mut();
    // SAFETY: `getifaddrs` writes an owned list to `head` on success (0). We
    // walk it without retaining pointers and hand it back to `freeifaddrs`.
    if unsafe { libc::getifaddrs(&mut head) } != 0 {
        return out;
    }
    let mut cursor = head;
    while !cursor.is_null() {
        // SAFETY: `cursor` is a non-null node of the list `getifaddrs` built,
        // valid until `freeifaddrs`.
        let entry = unsafe { &*cursor };
        if !entry.ifa_addr.is_null() {
            // SAFETY: `ifa_addr` is non-null and its `sa_family` tag selects
            // which concrete sockaddr the kernel stored, so each branch reads
            // the matching layout.
            let family = unsafe { (*entry.ifa_addr).sa_family } as i32;
            match family {
                libc::AF_INET => {
                    let sa = unsafe { &*(entry.ifa_addr as *const libc::sockaddr_in) };
                    out.push(IpAddr::from(u32::from_be(sa.sin_addr.s_addr).to_be_bytes()));
                }
                libc::AF_INET6 => {
                    let sa = unsafe { &*(entry.ifa_addr as *const libc::sockaddr_in6) };
                    let ip = std::net::Ipv6Addr::from(sa.sin6_addr.s6_addr);
                    // Link-local addresses need a scope id to be dialable, which
                    // a bare host:port string cannot carry. Listing them would
                    // hand out URLs that do not connect.
                    if !is_unicast_link_local(&ip) {
                        out.push(IpAddr::from(ip));
                    }
                }
                _ => {}
            }
        }
        cursor = entry.ifa_next;
    }
    // SAFETY: `head` is the list head returned by a successful `getifaddrs`,
    // freed exactly once here; no node pointer outlives this call.
    unsafe { libc::freeifaddrs(head) };
    out
}

/// `Ipv6Addr::is_unicast_link_local` is still unstable, so match `fe80::/10`
/// directly.
fn is_unicast_link_local(ip: &std::net::Ipv6Addr) -> bool {
    let octets = ip.octets();
    octets[0] == 0xfe && (octets[1] & 0xc0) == 0x80
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_bind_lists_only_itself() {
        let info = BindInfo::capture("127.0.0.1:11435".parse().unwrap());
        assert!(!info.is_wildcard());
        assert_eq!(info.addresses, vec!["127.0.0.1:11435".parse().unwrap()]);
        assert_eq!(info.to_json()["wildcard"], json!(false));
        assert_eq!(info.to_json()["port"], json!(11435));
    }

    #[test]
    fn wildcard_bind_enumerates_reachable_addresses() {
        let info = BindInfo::capture("0.0.0.0:11435".parse().unwrap());
        assert!(info.is_wildcard());
        assert_eq!(info.to_json()["host"], json!("0.0.0.0"));
        // Every listed address must be dialable: right family, right port, and
        // never the wildcard itself.
        for addr in &info.addresses {
            assert!(addr.is_ipv4());
            assert_eq!(addr.port(), 11435);
            assert!(!addr.ip().is_unspecified());
        }
        // Loopback exists on any host that can run the server, so a v4 wildcard
        // bind must enumerate at least that.
        assert!(info.addresses.iter().any(|addr| addr.ip().is_loopback()));
    }

    #[test]
    fn link_local_v6_is_excluded_because_it_needs_a_scope_id() {
        assert!(is_unicast_link_local(&"fe80::1".parse().unwrap()));
        assert!(is_unicast_link_local(&"febf::1".parse().unwrap()));
        assert!(!is_unicast_link_local(&"fec0::1".parse().unwrap()));
        assert!(!is_unicast_link_local(&"::1".parse().unwrap()));
    }
}
