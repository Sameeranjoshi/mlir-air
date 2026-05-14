*(Original URL: https://sdk.cerebras.net/csl/tutorials/gemv-05-multiple-pes/running.html)*

---

We add one additional compile time parameter to specify the width of our
program rectangle:

``` bash
$ cslc layout.csl --fabric-dims=11,3 --fabric-offsets=4,1 --params=M:4,N:6,width:4 --memcpy --channels=1 -o out
$ cs_python run.py --name out
```

We use the same command to run. You should see a `SUCCESS!` message at
the end of execution.
