# Deprecations

What Tack has deprecated: each still works as before, raises a
`tack.TackDeprecationWarning` (a `DeprecationWarning`) at the line that uses it,
and goes in the release named.

<!-- deprecations -->

Something is deprecated in one minor release and removed in the next or a
later one. Before 1.0, a breaking change may still go without a deprecation
where keeping the old behaviour would cost more than it saves; the
[CHANGELOG](https://github.com/Kitware/tack/blob/main/CHANGELOG.md) says so
when it does. Each deprecation is listed there under *Deprecated* in the
release that made it.

Python shows a `DeprecationWarning` in a script's own code and under pytest,
and hides it in libraries' code. To see every one, or to make them errors
while moving off them:

```python
import warnings
import tack

warnings.simplefilter("always", tack.TackDeprecationWarning)   # every time
warnings.simplefilter("error", tack.TackDeprecationWarning)    # as errors
```
