# Install Hypre

1. In the `src` directory, type `configure --prefix=/usr/local`.
2. Type `make install`.

# Compile with Hypre

```mpicc -I/usr/local/include {FILENAME}.c /usr/local/lib/libHYPRE.a -lm```
