//===- ChipletTypes.td - Chiplet types ---------------------*- tablegen -*-===//
#ifndef CHIPLET_TYPES
#define CHIPLET_TYPES

include "chiplet/Dialect/ChipletDialect.td"

//===----------------------------------------------------------------------===//
// !chiplet.event — opaque handle for cross-task dependencies.
//===----------------------------------------------------------------------===//

def Chiplet_EventType : Chiplet_Type<"Event", "event"> {
  let summary = "Opaque event handle for cross-task synchronization";
  let description = [{
    `!chiplet.event` represents a Fleet-style event counter used to signal
    completion of a chiplet/device task and to gate downstream tasks. Lowered
    in a later sub-project; here it is an opaque token type.
  }];
}

#endif // CHIPLET_TYPES
