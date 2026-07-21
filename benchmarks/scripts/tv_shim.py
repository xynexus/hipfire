"""Shim the broken torchvision namespace stub on nix1 so transformers imports.
Import this BEFORE importing transformers."""
import sys, types, importlib.machinery

def install():
    if 'torchvision.transforms.functional' in sys.modules:
        return
    try:
        import torchvision.io  # noqa
        import torchvision.transforms.functional  # noqa
        return  # real torchvision present and working; do not shim
    except Exception:
        pass
    tv = types.ModuleType('torchvision'); tv.__path__ = []
    tv.__spec__ = importlib.machinery.ModuleSpec('torchvision', None)
    tv.__version__ = '0.0.0-shim'
    io = types.ModuleType('torchvision.io')
    class ImageReadMode: pass
    def decode_image(*a, **k): raise NotImplementedError
    io.ImageReadMode = ImageReadMode; io.decode_image = decode_image
    tr = types.ModuleType('torchvision.transforms')
    class InterpolationMode:
        NEAREST='nearest'; NEAREST_EXACT='nearest-exact'; BOX='box'
        BILINEAR='bilinear'; HAMMING='hamming'; BICUBIC='bicubic'; LANCZOS='lanczos'
    tr.InterpolationMode = InterpolationMode
    fn = types.ModuleType('torchvision.transforms.functional')
    def pil_to_tensor(*a, **k): raise NotImplementedError
    fn.pil_to_tensor = pil_to_tensor
    tr.functional = fn; tv.io = io; tv.transforms = tr
    sys.modules['torchvision'] = tv
    sys.modules['torchvision.io'] = io
    sys.modules['torchvision.transforms'] = tr
    sys.modules['torchvision.transforms.functional'] = fn

install()
