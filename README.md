# Dependencies

## Hypre

1. Download Hypre from https://github.com/hypre-space/hypre.
1. In the `src` directory, type `./configure --prefix={PREFIX}`.
1. Type `make install`.

### Compile with Hypre

Add the include path `-I{PREFIX}/include` and the library path `-L{PREFIX}/lib`, then link Hypre with `-lHYPRE`.

## GLib

Installed on most Linux distro's by default. The header files usually can be found in `/usr/local/include`, `/usr/include/glib-2.0`, or `/usr/lib/glib-2.0/include`.

### Compile with GLib

Link with `-lglib-2.0`. If failed, please specify the include path manually.

## netCDF

Carefully follow the instructions in https://www.unidata.ucar.edu/software/netcdf/docs/getting_and_building_netcdf.html. The instructions set install path to `/usr/local` (`--prefix=/usr/local` in configure option); change in case of need.

### Compile with netCDF

Same with Hypre but link with `-lnetcdf`.
