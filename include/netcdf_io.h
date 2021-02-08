#ifndef NETCDF_IO_H
#define NETCDF_IO_H

#include <mpi.h>

#define NETCDF_MAX_LEN 100

typedef struct _solver Solver;

typedef struct _netcdf_reader {
    char filename[NETCDF_MAX_LEN];
    int Nx, Ny, Nz;
    double *xf, *yf, *zf;
} NetCDFReader;

NetCDFReader *NetCDFReader_Create(const char *filename);
void NetCDFReader_Destroy(NetCDFReader *reader);

typedef struct _netcdf_writer {
    char filename[NETCDF_MAX_LEN];
} NetCDFWriter;

NetCDFWriter *NetCDFWriter_Create(const char *filename);
void NetCDFWriter_Destroy(NetCDFWriter *writer);

#endif
