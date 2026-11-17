# Inspecting device-class objects

**Status: done.** Issue [#324](https://github.com/Kitware/tack/issues/324),
2026-11-14. Carried out in [#457](https://github.com/Kitware/tack/pull/457),
merged 2026-11-15; see [How it came out](#how-it-came-out).

## Summary

A kernel that takes a device-class object compiles what the object brings:
its class constants, its arrays and run-time scalars, the templates it holds
and the `@tack.func` methods found along its class's method resolution order.
Nothing shows that today. Someone extending a filter cannot ask which class
supplies a dataset view's `corner_value`, which of an object's numbers are
compiled in, or what an implicit array holds, short of reading the classes
and the template rewrite.

This plan adds one function, `tack.inspect_template(obj, name=None)`, in
tack-core. It describes an object that already exists, a user's template, an
implicit array or a dataset view, as a string, the way `tack.inspect` gives a
kernel's generated code. Given a method name it describes that method. It
uses the compiler's own rules for classifying an object, so the two cannot
disagree, and it does nothing that dispatch would notice: no device reads,
launches, compilation, disk-cache lookups, backend start-up or cache entries.

A companion, specialization diagnostics
([#325](https://github.com/Kitware/tack/issues/325)), follows it: this plan
says what an object supplies, that one why calls compiled the variants they
did.

## Where things stand

On `main` at `2d1325bd`:

- **Classification** is in `lang/template_rewrite.py`.
  `classify_template_attrs(obj)` flattens an object's tree into compile-time
  constants, arrays and run-time scalars (nested names joined by `__`, tuple
  entries named `shape__0`), `template_tree` lists the tree,
  `template_func_methods(cls)` collects the `@tack.func` methods along the
  method resolution order (unwrapping `@staticmethod`), and
  `template_func_attrs(obj)` the device functions held as instance
  attributes.
- **Two of these leave traces.** `_class_constants(cls)` records each class's
  constant names in `_CONSTANT_NAMES` the first time it is asked, and keeps
  them: a numeric attribute added to the class after that is not a constant.
  `_classify_own` reads instance attributes with `getattr`, which runs a
  property a class defines over an instance attribute's name. Inside
  `template_scope` all of them are memoized by object id.
- **Every `Func` has its source.** `Func.__init__` reads and parses it, and
  raises if it cannot, so a method's source text, parsed tree, file and line
  are always there (`Func.func`, `Func._funcdef`). The proposal's "source
  unavailable" case does not arise for device functions.
- **Methods reach one another in four ways**, all resolved by the template
  rewrite: `self.m(...)` (and `self.base.m(...)` through held templates),
  a device function held as an attribute (`self.kernel(r, h)`), the indexing
  protocol (`self[i]`, `self[i] = v` and `len(self)` become `__getitem__`,
  `__setitem__` and `__len__`, and `self[i] += v` both of the first two;
  `_Indexing`), and a template passed to a free `@tack.func` (`helper(self)`,
  `_Specializer`).
- **Free names resolve through `CallBindings`** (`lang/call_bindings.py`).
  `resolve` looks a name up in the function's closure and globals and
  reports the answer to the disk cache's recorder when one is active.
  `_resolve` does the lookup, but for a module-qualified name
  (`helpers.sample`) it finds the module through `resolve`, so the recorder
  hears of it too.
- **Dataset views** are classes made by `views.view_class(kind, shape,
  *mixins)` (tack-vis), once per `(kind, shape, mixins)`, after
  `_check_names` refuses two parts that define one name differently. So every
  method of a view has one defining class. The parts are themselves often
  made per order (`of_order`, `with_table`, `of_node_order`). A view is
  obtained as before: `data.launch_groups(domain, fields)`, then
  `data.domain_view(domain, group)` and `field.view(group)`.
- **Implicit arrays** (`tack.arrays`) are device classes whose arrays and
  scalars are instance attributes, and whose sizes and dtypes are underscored
  metadata that kernels do not receive.
- `tack.inspect(kernel, *args, mode=...)` is in `lang/inspect_kernel.py`,
  documented in `docs/reference/api.md` ("Inspection") and
  `docs/contracts/runtime-api.md`.

The design was proposed and reviewed on 2026-10-31 against `c5a015c0`. Since
then `tack.arrays` moved out of `tack.dataset.arrays`, indexable objects and
host implementations (`Func.host`) arrived, and views gained the polyhedral,
polygon, polyline, wedge, simplex and tensor parts. None of that changes the
design; the four ways of reaching a method and the cached constant names are
what this version adds.

## D1. One function, returning a string

**Recommendation:** `tack.inspect_template(obj, name=None) -> str`, exported
from `tack` and listed in `__all__`.

```python
print(tack.inspect_template(my_template))
print(tack.inspect_template(my_template, "sample"))
print(tack.inspect_template(tack.arrays.CountingArray(10)))

temperature = data.fields["T"]
group = next(g for g in data.launch_groups("cells", [temperature]) if g.count)
print(tack.inspect_template(data.domain_view("cells", group)))
print(tack.inspect_template(temperature.view(group), "corner_value"))
```

- `obj` is an instance of a device class. Anything else raises `TypeError`,
  a class included ("pass an instance: what is compiled depends on its
  attributes").
- `name` is a method of the object's class, by its Python name. An unknown
  name raises `AttributeError`; a member that is not a device function (a
  constant, an array) raises `TypeError` naming what it is. Dotted names
  (`"base.get"`) are not taken: nested objects' methods are in the full
  report.
- The text is for people. Its layout, internal class names and line numbers
  are not a stable interface, as `tack.inspect`'s source is not. A report
  object or JSON waits for something that needs one.

The alternatives were a dataset-only `td.inspect_view`, which the review
dropped since a view is a device class like any other, and a report class,
deferred.

## D2. The compiler's rules, read without side effects

**Recommendation:** factor classification so that lowering and inspection
call the same code, and let inspection read it without leaving traces.

- **Constant names.** Split `_class_constants` into a pure scan of the
  method resolution order (`vars(klass)` of each class) and the cache.
  Inspection uses the cached names when the class has them (what lowering
  uses from then on) and otherwise scans without storing.
- **Instance attributes.** `_classify_own` takes the attribute values as an
  argument. Lowering passes `getattr`'s, as now. Inspection passes
  `vars(obj)`'s, which run nothing, and, where the class defines a data
  descriptor (a property) over an attribute's name, shows that attribute as
  "not read: a property of the class", since lowering's value would come
  from running it.
- **Methods.** `template_func_methods` gains the class that defines each
  method (the walk already passes it), so the report and lowering find
  methods the same way.
- **No memo, no scope.** Inspection does not enter `template_scope`, so the
  per-object memo is untouched.
- **Cycles and collisions.** A tree that holds itself, or two attributes
  flattening to one name, are reported in the text where they occur, with
  the message lowering would raise, rather than raising.

A test checks the equivalence directly: for every device class the suite's
template tests define, inspection's constants, arrays and run-time scalars
equal `classify_template_attrs`' (by name and kind), and lowering afterwards
finds the same `_CONSTANT_NAMES` as without the inspection.

The alternative, an inspector with its own classification, would drift from
the compiler the first time either changed.

## D3. What the full report shows

**Recommendation:**

1. The class: module and qualified name, its bases and method resolution
   order, and for a dataset view its composition (see D5).
2. **Compiled in:** class constants with their values and the class that
   defines each (an instance number overriding one is marked, since it stays
   compiled in); constant tuple entries as one line per tuple.
3. **Passed at run time:** instance scalars and tuples of numbers.
4. **Arrays:** name, logical shape, dtype, vector or matrix width, layout
   (`aos`/`soa`), and the device as recorded on the array (`cuda:0`), read
   from the backend it holds without making a `Device` or selecting a
   default. Never values, bounds or addresses.
5. **Held objects:** each nested template under its path, described the same
   way and indented. An object reached twice is described once and named
   where it recurs ("the same object as `base`"). Implicit arrays are
   described as what they hold, never materialized.
6. **Device functions held as attributes**, with what each is.
7. **Methods:** each `@tack.func` by name with its defining class and
   `file:line`; protocol methods (`__getitem__`, `__setitem__`, `__len__`)
   say which syntax they serve (`obj[i]`, `obj[i] = v`, `len(obj)`), and a
   method with a host implementation says so. `__tack_iterate__`, where
   set, is shown with the loop it makes.

Output is bounded: a constant tuple shows its first eight entries and a
count, a long method list stays whole (it is what was asked for), and
nesting deeper than eight levels is cut with a count of what was left out.
Values are printed by Tack, never through an attribute's `repr`.

## D4. Asking about one method

**Recommendation:** `tack.inspect_template(obj, "corner_value")` gives

1. the method and the class that defines it, along the method resolution
   order;
2. its module, qualified name and `file:line`;
3. its signature as written;
4. the calls in its source that reach another method or device function, in
   source order, each with what it resolves to:
   - `self.m(...)`, and `self.a.b.m(...)` through held templates, to the
     defining class of `m` on that object;
   - `self.f(...)` where `f` is a held device function, to that function;
   - `self[i]`, `self[i] = v` and `len(self)` (also on `self.a`), to
     `__getitem__`, `__setitem__` and `__len__`, and `self[i] += v` to both
     `__getitem__` and `__setitem__`, as `_Indexing` rewrites it; "not
     defined" where the template rewrite would emit its refusal;
   - a free name called with `self` or `self.a` among its arguments, plain
     (`helper(self)`) or module-qualified (`helpers.sample(self)`), to that
     `@tack.func`. The name is found by a new `CallBindings.lookup(node)`,
     which applies `resolve`'s rules recursing through itself and reports to
     no recorder; `resolve` keeps its recording, and the two share the step
     that resolves one node, so they cannot come to differ.

For an order-1 H1 field view on `main` (lines are each `def`'s):

```text
corner_value: defined by _Interpolated
source: packages/tack-vis/src/tack/dataset/views.py:1400
signature: corner_value(self, i, j)
calls in its source:
  self.dof(i, j)    -> _H1Field.dof    views.py:1432
```

The list is read from the source text, not from what lowering inlines: a
call in a branch constants decide against is listed, a call through a local
alias (`f = self.m; f(x)`) is listed as unresolved, and nothing is executed,
lowered or followed further. It answers "where do I look next", not "what
does the kernel run".

The alternative, the transitive set of methods lowering would inline, needs
lowering itself (or a second implementation of its decisions), and its
answer depends on the kernel and its arguments.

## D5. A view's composition, recorded where it is made

**Recommendation:** `views.view_class` stores its key on the class it makes,
as `cls._tack_composition = (kind, shape, mixins)`. The underscore keeps it
out of classification, and it is on the class, not the instance, so no view
holds a dataset. tack-core reads the attribute by name and does not import
tack-vis.

The report shows it as the domain kind, the shape and the parts in order,
with each part's own name (`_H1TensorField` of order 3), before the method
resolution order. Methods already name their defining class, which with
`_check_names` is the one part that defines them.

Nothing is reconstructed: inspection describes the view it is given, and
does not rerun the space's attribute logic or group splitting.

## D6. Delivery

**Recommendation:** one pull request, three commits, since the whole is a
few hundred lines and the parts are reviewed together:

1. The shared, side-effect-free classification (D2) and the full report
   (D1, D3), with its tests, the API page's entry and the contract's.
2. The one-method query (D4), with its tests.
3. The view composition (D5), an example in the views guide, the device
   classes guide's section, the developer guide's templates chapter, and the
   changelog.

The alternative is the proposal's three pull requests, one per step.

## Tests

In tack-core (`test_inspect_template.py`, quick tier):

- Objects: a user template with class constants, an instance number
  overriding one, instance scalars and tuples, arrays (vector, matrix, SOA),
  nested templates two levels deep, an object reached twice, a held device
  function, inherited, overridden and static methods, protocol methods, a
  host implementation, `__tack_iterate__`, and implicit arrays
  (`CountingArray`, `Permutation` over an array, `Concatenate`).
- Classification equals `classify_template_attrs`' for every device class
  the template test modules define (D2).
- No side effects: device reads, launches, compilation, disk-cache lookups
  and backend creation patched to fail; a property over an attribute's name
  not run; `_CONSTANT_NAMES`, the template memo, variant counts and the
  dispatch fast path unchanged; the returned string holds no reference to
  the object, its class or its arrays (a weak reference dies).
- Errors: a non-device-class object, a class, an unknown method, a member
  that is not a device function; a cycle and a name collision reported in
  the text.
- The one-method query: each of D4's four kinds of call, including
  `self[i] += v` and `self.a[i] += v` (both protocol methods listed) and a
  module-qualified free function; a call in a constant-decided branch, an
  unresolved alias, a free function not bound.
- No recording: a method query under an active disk-cache recorder, with a
  plain and a module-qualified free call, leaves the recorder's notes as they
  were; `lookup` and `resolve` answer alike for every name the test modules'
  device functions call.
- Bounded output for a long constant tuple and deep nesting.

In tack-vis (`test_inspect_views.py`): a domain and a field view of
structured, unstructured, mixed-shape, selected, polyhedral, polygon and
polyline meshes; varying-order and high-order (tensor, simplex, wedge,
table-backed) fields; implicit and SOA storage; a node view. Each shows its
composition, and each method asked about names one defining class. The
methods asked about are ones each view has, not one list for all.

All CPU: inspection touches no device, which the patched device reads
check. A view of GPU arrays is shown the same way, and needs no hardware run
of its own.

## How it came out

[#457](https://github.com/Kitware/tack/pull/457) carried out D1 to D5 as
recommended: `tack.inspect_template` in `lang/inspect_template.py`, the
classification shared with lowering, `template_func_owners`,
`CallBindings.lookup`, and `_tack_composition` on view classes. The tests
are 29 in tack-core and 18 in tack-vis, all in the quick tier.

**Where it departs from the plan:**

- **D6:** two commits for the work, not three. The one-method query shares the
  report's module and helpers, so it was not split out. Two more commits came
  from review (below).
- **Tests, D2:** the equivalence with `classify_template_attrs` is checked on
  six fixed objects (a user template with nested objects, a leaf,
  `CountingArray`, `Permutation`, `Concatenate`, `ConstantArray`), not on
  every device class the template test modules define.
- **Tests, views:** there is no points domain, so face and edge domain views
  stand in for the "node view". A table-backed H1 field view is covered.
- **D4's example:** a source location is given from the top-level package
  (`tack/dataset/views.py:1400`), not from the repository root.

**What review added.** The review of #457 found three places where inspection
could disagree with lowering, or run code. Each is fixed, with a test that
runs the kernel or the compiler's classification first:

- **A held device function shadows a method of its name,** as lowering's
  `_template_nodes` has it. D4's call targets and the method query follow
  that, and the methods list marks a shadowed method "not used".
- **`len(obj)` is `obj.__len__()` whatever `len` is bound to,** since
  `_Indexing` rewrites it from the syntax alone.
- **No metaclass code runs (D2, extended).** Class values (the constants,
  `__tack_iterate__`, the `_device_class` mark, the property checks) are read
  from the classes' dictionaries (`_static`, `_is_template_static`,
  `iteration_space(cls, read)`), never with `getattr`. A value Python would
  run code for is "not read", and a metaclass that defines
  `__getattribute__` is named in the report, since lowering reads class values
  through it. Class metadata (`__qualname__`, `__mro__`) is still read through
  Python's attribute lookup. The flag that asks for all this is
  `inspecting=True` (D2's "scan without storing").

## Not in this plan

- Structured output (a report class, JSON), and a combined browser of
  objects and variants.
- Dotted method names.
- The compile log and key components: [#325](https://github.com/Kitware/tack/issues/325),
  which will link this function from its documentation.
