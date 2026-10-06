# Regenerate the expected renders with Python jinja2, configured the way HuggingFace's
# apply_chat_template configures it (trim_blocks, lstrip_blocks, loopcontrols, its
# tojson and raise_exception). hipfire's minijinja render must match them byte for
# byte -- see `qwen_templates_render_like_the_reference` in src/lib.rs.
#
#   python3 crates/hipfire-prompt/tests/fixtures/qwen-templates/render_reference.py
import json, os
import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

HERE = os.path.dirname(os.path.abspath(__file__))

def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)

def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

def render(template, conv, think, effort):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    kwargs = {"reasoning_effort": effort} if effort else {}
    return env.from_string(template).render(
        messages=conv["messages"], tools=conv["tools"], add_generation_prompt=True,
        enable_thinking=think, **kwargs)

conv = json.load(open(os.path.join(HERE, "conversation.json")))
CASES = [("qwen3.8-27b", False, None), ("qwen3.8-27b", True, None), ("qwen3.8-27b", True, "low"),
         ("qwen3.6-35b-a3b", False, None), ("qwen3.6-35b-a3b", True, None)]
for model, think, effort in CASES:
    template = open(os.path.join(HERE, f"{model}.jinja")).read()
    name = f"expected-{model}-{'think' if think else 'nothink'}{'-' + effort if effort else ''}.txt"
    with open(os.path.join(HERE, name), "w") as f:
        f.write(render(template, conv, think, effort))
    print(name)
