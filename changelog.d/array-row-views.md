### Changed

- Many views of one array's consecutive rows are made in one pass
  (`Array._row_views`, private: the view class and the root's fields worked out
  once), as an AMR hierarchy's boxes' fields and marks are: 2.2 us a view
  against 2.7 by slicing on the CPU.
