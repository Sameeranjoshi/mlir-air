2 approaches :
a) SPADA/AIE like
	Data and functions are associated with the hardware tiles, so think like a big bucket of shared data(D1, D2, D2,...) and functions(F1, F2, F3, ...) and some set of tiles(T1, T2, T3, ...), we associate data and functions with tiles.
	SPADA  - does a more general way using loops 
	AIR - does a manual way using indexes.(AIE hardware has only a few cores, say 32 (8x4) or similar but cerebras has a million pes, so we definitely need a SPADA like or some look like or some form where we don't do a pin point mapping)

B) CSL/Stencils-MLIR paper like
	Define a single package, say package (P1, P2, ...), each package has it's own data, functions, tasks, ... and set of tiles(T1, T2, T3,...), later in layout mapping or somewhere in csl.place place these Packages onto tiles.
	example csl.place(P1) onto T1 and so on.
	Seems like this is the current CSL language progamming approach, this is called ACTOR model, meaning that you define actors and they are assigned some location, this is a famous model for concurrency and distributed programming. actors talk using queues, and messages.
	[https://en.wikipedia.org/wiki/Actor_model](https://en.wikipedia.org/wiki/Actor_model)
	[https://www.theserverside.com/blog/Coffee-Talk-Java-News-Stories-and-Opinions/How-the-Actor-Model-works-by-example](https://www.theserverside.com/blog/Coffee-Talk-Java-News-Stories-and-Opinions/How-the-Actor-Model-works-by-example)
	[https://doc.akka.io/libraries/akka-core/current/typed/guide/actors-intro.html](https://doc.akka.io/libraries/akka-core/current/typed/guide/actors-intro.html)

C) What else? Which is a good decision here? Or do we need hybrid with good from both worlds? 

- I agree with "A couple of dimensions all three sketches share" topic. 
- sdkLayout APIs is a way in runtime calls to get rid of colorings and layout, think of it as a layout compiler.
[https://sdk.cerebras.net/api-docs/sdklayout-api](https://sdk.cerebras.net/api-docs/sdklayout-api)
- Let's concretize CSL dialect(and runtime and layout) and it's translation to actual CSL code first before we start doing AIR to CSL, as CSL is a base if we test, and implement it well, and understand the design well, the rest is just conversions from AIR to CSL.
- Opinions on option B:
Line "I would not pick Sketch B.", and "The PE-as-unit model is right for AIE because each AIE core has its own ELF; it's wrong for CSL because each PE source file is shared.", for CSL as well there will be final each core having it's elf, but yes having a csl file for each PE is the abstraction we look at, later csl compiler 'cslc' does the job of binary. 
- I check there are a lot of ops in the csl and csl_rt dialect as of now, seems like a lot of boilerplate, AIE does in a few essential operations, so many runtime calls/apis can be inferred/templatized, say somehow we can infer from mlir that the type of our variable used in code and we can spit export_name and such constructs, this was example not sure if true. 
- think about a template + constructs/ops approach if needed and see what can be possible or not. 
- For now focus on only functions(tasks, async, inter PE can be though step by step, keep first simple and robust dialect to build confidence)
- No DSD's for first version but they should be inferred automatically, say if there is a vector operation.
*comptime block arguments seem good idea.
- some structure in mind, might be not standard how upstream dev write code, now this has various ways one can think, the one below is a pure actor model discussed in approach B in point at starting. rough sketch, not necessarily we want to go with this, depends on what we decide as answer to the first point.

csl.wafer(@WSE3){
	// hardware resources section, global and common across a wafer
	%PE1 = csl.pe(0,0)	// pe(cols, rows)
	%PE2 = csl.pe(0,1)
	%PE3 = csl.pe(0,2)
	// other resources

```
csl.program(%arg1: type, %arg2: type){	// takes compile time comptime block arguments which convert into params(if they make sense and map well to CSL)
	// Mostly upstream MLIR core dialects, arith, loops(scf), ... 
	// performs various operations
	// can have csl.func or csl.task
	csl.func(){}
	csl.func(){}
	csl.func(){}
	... 
	csl.task(){}
	csl.task(){}
	...
	//some form of state machine or scheduling task which say run f1, then f2 and then this and then that and so on ... 
}

csl.program(){

}
... other programs


csl.layout(){
	// placing programs onto PEs
}
csl.host(){
	// host runtime code
}
```

}

- What's the exact AIRToCSLLayout / AIRToCSLProgram / AIRToCSLHost split? --> how does AIE think on this? Maybe we should follow AIE? because at some point I want to switch to using [https://sdk.cerebras.net/api-docs/sdklayout-api](https://sdk.cerebras.net/api-docs/sdklayout-api) this API is python way for layout file and layout dialect, so ya think on this.
- One liner is good air-translate --emit-csl-rt
- Out of scope for this doc, but flagging: --> we follow what AIR does, I think it knows it I guess, but ideally the plan is AIR will have a layer on top of it, which will tell where is place, route and ... done using a scheduling language.
- One way to look at how AIE programs are written here's a general outline:
  - hardware resources(tile, buffer, color channels, locks, ... ) -- limited resources/need to allocate/or ask for resources
  - routing decided/laid out on the fabric/chip
  - kernel scope(aie.core)
  - runtime secquence
- 

