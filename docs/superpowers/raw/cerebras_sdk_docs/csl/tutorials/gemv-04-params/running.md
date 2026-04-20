*(Original URL: https://sdk.cerebras.net/csl/tutorials/gemv-04-params/running.html)*

---

We\'ve shown how the device and host code use the compile time
parameters, but how do we set them? Our compile command now includes a
`--params` flag, which specifies the values:

``` bash
$ cslc layout.csl --fabric-dims=8,3 --fabric-offsets=4,1 --params=M:4,N:6 --memcpy --channels=1 -o out
$ cs_python run.py --name out
```

We use the same command to run. You should see a `SUCCESS!` message at
the end of execution.
