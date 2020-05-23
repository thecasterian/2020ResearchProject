# Install Hypre

1. In the `src` directory, type `configure --prefix=/usr/apps`.
2. Type `make install`.

# Compile with Hypre

```mpicc -I/usr/apps/include {FILENAME}.c /usr/apps/lib/libHYPRE.a -lm```
