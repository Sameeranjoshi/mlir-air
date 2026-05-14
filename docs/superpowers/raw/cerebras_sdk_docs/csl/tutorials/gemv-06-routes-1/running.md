*(Original URL: https://sdk.cerebras.net/csl/tutorials/gemv-06-routes-1/running.html)*

---

Since this program only uses two PEs, we adjust our simulated fabric
dimensions accordingly:

``` bash
$ cslc layout.csl --fabric-dims=9,3 --fabric-offsets=4,1 --params=M:4,N:6 --memcpy --channels=1 -o out
$ cs_python run.py --name out
```

We use the same command to run. You should see a `SUCCESS!` message at
the end of execution.
