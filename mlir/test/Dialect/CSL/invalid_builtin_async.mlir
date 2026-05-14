// RUN: not air-opt %s 2>&1 | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p {
    %a = csl.var @a : memref<4xf32>
    csl.func @c {
      %ad = csl.get_mem_dsd %a : memref<4xf32> -> !csl.dsd
      // CHECK: error: 'csl.builtin_call' op 'activate' requires 'async'
      csl.builtin_call "fmovs"(%ad, %ad) {activate = @done}
        : (!csl.dsd, !csl.dsd) -> ()
      csl.return
    }
    csl.task @done attributes {trigger_kind = "local_task_id", id = 1 : i32} { csl.return }
  }
  csl.layout {width = 1, height = 1} @layout { csl_layout.place @p at (0,0) }
}
