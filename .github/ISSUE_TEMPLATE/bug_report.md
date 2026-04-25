---
name: Bug report
about: Something isn't working — a kernel produces wrong output, an install fails, a demo crashes
title: '[bug] '
labels: bug
assignees: ''
---

## What happened

<!-- A clear, one-paragraph description of what went wrong. -->

## Minimal reproduction

<!--
Ideally 5-10 lines of code that reproduces the issue.
If the bug is install-related, paste the exact `pip install` command
and the failing output instead.
-->

```python
import torch
import sparselab

# paste repro here
```

## What you expected

<!-- What should have happened instead? -->

## What actually happened

<!--
Paste the full traceback or error output. If the bug is silent
(wrong numbers, not a crash), paste the actual output and what
would have been correct.
-->

```
<traceback or error output here>
```

## Environment

Please run these commands and paste the output:

```bash
python --version
python -c "import platform; print(platform.machine(), platform.platform())"
python -c "import sparselab; print('sparselab', sparselab.__version__)"
python -c "import torch; print('torch', torch.__version__)"
pip --version
```

Paste output here:

```
<output here>
```

## Additional context

<!-- Anything else — screenshots, related issues, your hypothesis about the cause. -->
