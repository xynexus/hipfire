---
name: Hipfire Admin Console
description: A quiet, technical control plane for trusted inference operations.
colors:
  canvas: "#111417"
  surface: "#171b20"
  text: "#e7ecef"
  muted: "#9aa5af"
  line: "#29313a"
  accent: "#2dd4bf"
  warning: "#f59e0b"
  danger: "#ef4444"
typography:
  title:
    fontFamily: "Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "1.125rem"
    fontWeight: 650
    lineHeight: 1.3
  body:
    fontFamily: "Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "0.75rem"
    fontWeight: 600
    lineHeight: 1.3
  data:
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace"
    fontSize: "0.75rem"
    fontWeight: 400
    lineHeight: 1.4
rounded:
  control: "6px"
  surface: "10px"
  pill: "999px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "14px"
  lg: "20px"
  xl: "24px"
components:
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.canvas}"
    typography: "{typography.label}"
    rounded: "{rounded.control}"
    padding: "8px 12px"
  button-secondary:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text}"
    typography: "{typography.label}"
    rounded: "{rounded.control}"
    padding: "8px 12px"
  field:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.text}"
    typography: "{typography.body}"
    rounded: "{rounded.control}"
    padding: "8px 10px"
  panel:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text}"
    rounded: "{rounded.surface}"
    padding: "14px 16px"
---

# Design System: Hipfire Admin Console

## Overview

**Creative North Star: "Quiet Technical Trust"**

The console is a precise instrument used by an operator who may be responding
to load, access, or safety issues. Its restrained dark-neutral canvas reduces
glare, while a scarce teal accent identifies current selection, healthy state,
and primary action. Information is dense but never cramped; hierarchy comes
from alignment, type weight, and tonal layers rather than decoration.

The system rejects generic SaaS marketing dashboards, cyberpunk control rooms,
card walls, modal-first administration, and terminal cosplay. Familiar browser
controls and explicit language should disappear into the operator's task.

**Key Characteristics:**

- Restrained teal on cool charcoal neutrals.
- Compact, repeatable control and table vocabulary.
- Operational state and consequences shown in context.
- Responsive structure with keyboard-complete workflows.
- Motion limited to fast state feedback and always reduced-motion safe.

## Colors

The palette is a cool, low-chroma operational field with teal reserved for
meaningful state and action.

### Primary

- **Signal Teal:** The accent marks primary actions, selected navigation,
  healthy progress, and visible focus. It is never background decoration.

### Neutral

- **Deep Canvas:** The page background establishes the lowest tonal layer.
- **Instrument Surface:** Panels, rows, and controls use the raised neutral.
- **Clear Ink:** Primary text and essential values use the highest contrast.
- **Measured Muted:** Secondary labels use this only where contrast remains AA.
- **Structural Line:** Dividers and control outlines clarify grouping without
  turning every region into a card.
- **Amber Warning:** Recoverable degraded or pending states.
- **Direct Red:** Destructive controls and error states only.

**The Scarce Signal Rule.** Teal identifies action or state on no more than a
small minority of the screen; its rarity makes it trustworthy.

**The No Color-Only Rule.** Every warning, error, selection, and status also has
text, shape, or an accessible state that carries the same meaning.

## Typography

**Display Font:** Inter with the system UI stack
**Body Font:** Inter with the system UI stack
**Label/Mono Font:** System monospace for identifiers and measured values

**Character:** One practical sans family keeps labels, controls, and prose
coherent. Monospace appears only where fixed-width comparison improves scanning.

### Hierarchy

- **Title:** Semibold and compact; reserved for the page and major panes.
- **Headline:** Semibold; identifies a tab's primary working region.
- **Body:** Regular; used for explanations and control labels, with prose capped
  at 70 characters per line.
- **Label:** Semibold and compact; used for table headings, filters, and metadata
  without reflexive uppercase tracking.
- **Data:** Monospace; used for IDs, tokens, timestamps, limits, and quantities.

**The Instrument Label Rule.** Labels say what a value means; abbreviations and
uppercase are used only when they are established technical vocabulary.

## Elevation

The console is flat by default. Tonal layering and thin structural dividers
create depth; shadows are reserved for transient overlays that must sit above
the working surface, such as a confirmation dialog or menu.

**The Flat-by-Default Rule.** A resting panel never combines a border with a
wide decorative shadow. If every region appears to float, hierarchy has failed.

## Components

Components are restrained, familiar, and complete across default, hover,
focus-visible, active, disabled, loading, and error states.

### Buttons

- **Shape:** Gently squared controls with a 6px radius.
- **Primary:** Signal Teal with Deep Canvas text and compact 8px by 12px padding.
- **Hover / Focus:** A small tonal shift on hover and a clearly visible 2px focus
  outline with offset; state transitions finish within 200ms.
- **Secondary / Ghost:** Neutral surfaces or text-only actions; destructive
  variants use Direct Red plus explicit verbs.

### Chips

- **Style:** Compact neutral controls with a 6px radius and structural outline.
- **State:** Selected filters use teal text plus a persistent selected state;
  they do not rely on fill color alone.

### Cards / Containers

- **Corner Style:** Restrained 10px radius where a bounded surface is needed.
- **Background:** Instrument Surface over Deep Canvas.
- **Shadow Strategy:** None at rest.
- **Border:** Structural Line only when separation cannot be achieved by spacing
  or tonal contrast.
- **Internal Padding:** 14px by 16px for compact operational content.

### Inputs / Fields

- **Style:** Deep Canvas field, Structural Line outline, 6px radius, and a minimum
  44px touch target where the layout permits.
- **Focus:** Signal Teal outline with visible offset.
- **Error / Disabled:** Explicit adjacent message; disabled controls retain
  legible labels and expose their disabled state programmatically.

### Navigation

Overview, Access, and Usage use familiar tabs within one persistent console
shell. The active tab has text and shape reinforcement. At narrow widths the
tabs remain horizontally reachable and tables switch to labelled row layouts or
safe overflow without hiding actions.

## Do's and Don'ts

### Do:

- **Do** put user identity, token identity, scope, time range, and effective
  limit beside the data or action they qualify.
- **Do** use loading skeletons, explanatory empty states, and recoverable inline
  errors.
- **Do** give destructive actions explicit verbs, consequences, confirmation,
  and a stable focus return target.
- **Do** preserve keyboard order, visible focus, 44px touch targets, WCAG 2.2 AA
  contrast, and reduced-motion behavior.
- **Do** keep tokens and measured values in monospace while leaving prose and
  controls in the system sans.

### Don't:

- **Don't** build generic SaaS marketing dashboards with decorative hero
  metrics.
- **Don't** use cyberpunk control rooms with neon gradients, glass panels, or
  gratuitous animation.
- **Don't** create card walls that fragment related data into identical floating
  tiles.
- **Don't** hide destructive actions behind vague icon-only controls or
  ambiguous copy.
- **Don't** imitate dense terminals at the expense of hierarchy, accessibility,
  or familiar browser affordances.
- **Don't** use colored side-stripe borders, gradient text, glassmorphism,
  over-rounded surfaces, or a border paired with a wide decorative shadow.
