### Added

- `tack.inspect_template(obj, name=None)` describes a device-class object
  (yours, an implicit array of `tack.arrays`, a dataset view) as a kernel
  sees it, as a string: what is compiled in and what is passed at each call,
  its arrays, the objects and device functions it holds, and every
  `@tack.func` method with the class that defines it and its file and line.
  A dataset view also lists the parts it is made of. Given a method's name,
  it shows that method's defining class, source and signature, and the calls
  in its source to other methods and device functions, each with what it
  resolves to. It runs nothing: no property, device read, launch,
  compilation or disk-cache lookup (#324).
