module {
  %0 = csl_rt.create_layout : !csl_rt.layout
  %1 = csl_rt.create_code_region %0 "pe_program.csl", "main", 16 : index, 16 : index : !csl_rt.layout -> !csl_rt.code_region
  %2 = csl_rt.place %1 at(0 : index, 0 : index) : !csl_rt.code_region -> !csl_rt.code_region
  %3 = csl_rt.compile %0 : !csl_rt.layout -> !csl_rt.compile_artifacts
}

