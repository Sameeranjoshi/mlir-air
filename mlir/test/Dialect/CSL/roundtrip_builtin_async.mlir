// RUN: air-opt %s | air-opt | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p {
    %a = csl.var @a : memref<128xf32>
    csl.func @c {
      %ad = csl.get_mem_dsd %a : memref<128xf32> -> !csl.dsd
      %n = arith.constant 128 : index
      %fd = csl.get_fab_dsd fabout @send extent(%n : index) : !csl.dsd
      // CHECK: csl.builtin_call "fmovs"
      // CHECK-SAME: {activate = @done, async}
      csl.builtin_call "fmovs"(%fd, %ad)
        {async, activate = @done}
        : (!csl.dsd, !csl.dsd) -> ()
      csl.return
    }
    csl.task @done attributes {trigger_kind = "local_task_id", id = 8 : i32} { csl.return }
  }
  csl.layout {width = 1 : i64, height = 1 : i64} @layout {
    csl.color @send
    csl_layout.place @p at (0, 0)
  }
}
