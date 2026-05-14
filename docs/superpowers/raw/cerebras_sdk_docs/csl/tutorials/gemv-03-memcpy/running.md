*(Original URL: https://sdk.cerebras.net/csl/tutorials/gemv-03-memcpy/running.html)*

---

As with the previous tutorial, we compile and run this code using:

``` bash
$ cslc layout.csl --fabric-dims=8,3 --fabric-offsets=4,1 --memcpy --channels=1 -o out
$ cs_python run.py --name out
```

You should see a `SUCCESS!` message at the end of execution.
