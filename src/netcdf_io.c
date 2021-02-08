#include "../include/netcdf_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <netcdf.h>

NetCDFReader *NetCDFReader_Create(const char *filename) {
    NetCDFReader *reader;
    int ncid;
    int x_dimid, y_dimid, z_dimid;
    size_t x_dimlen, y_dimlen, z_dimlen;
    int xf_varid, yf_varid, zf_varid;
    int stat;

    reader = calloc(1, sizeof(*reader));
    snprintf(reader->filename, NETCDF_MAX_LEN, "%s.nc", filename);

    /* Open file. */
    stat = nc_open(reader->filename, NC_NOWRITE, &ncid);
    if (stat != NC_NOERR) {
        printf("error: cannot open file %s\n", reader->filename);
        goto error;
    }

    /* Read dimensions. */
    stat = nc_inq_dimid(ncid, "x", &x_dimid);
    if (stat != NC_NOERR) {
        printf("error: dimension 'x' not found\n");
        goto error;
    }
    stat = nc_inq_dimid(ncid, "y", &y_dimid);
    if (stat != NC_NOERR) {
        printf("error: dimension 'y' not found\n");
        goto error;
    }
    stat = nc_inq_dimid(ncid, "z", &z_dimid);
    if (stat != NC_NOERR) {
        printf("error: dimension 'z' not found\n");
        goto error;
    }

    nc_inq_dimlen(ncid, x_dimid, &x_dimlen);
    nc_inq_dimlen(ncid, y_dimid, &y_dimlen);
    nc_inq_dimlen(ncid, z_dimid, &z_dimlen);

    reader->Nx = x_dimlen;
    reader->Ny = y_dimlen;
    reader->Nz = z_dimlen;

    /* Read coordinates. */
    stat = nc_inq_varid(ncid, "xf", &xf_varid);
    if (stat != NC_NOERR) {
        printf("error: variable 'xf' not found\n");
        goto error;
    }
    stat = nc_inq_varid(ncid, "yf", &yf_varid);
    if (stat != NC_NOERR) {
        printf("error: variable 'yf' not found\n");
        goto error;
    }
    stat = nc_inq_varid(ncid, "zf", &zf_varid);
    if (stat != NC_NOERR) {
        printf("error: variable 'zf' not found\n");
        goto error;
    }

    reader->xf = calloc(x_dimlen+1, sizeof(double));
    reader->yf = calloc(y_dimlen+1, sizeof(double));
    reader->zf = calloc(z_dimlen+1, sizeof(double));

    nc_get_var_double(ncid, xf_varid, reader->xf);
    nc_get_var_double(ncid, yf_varid, reader->yf);
    nc_get_var_double(ncid, zf_varid, reader->zf);

    /* Close file. */
    nc_close(ncid);

    return reader;

error:
    free(reader);
    return NULL;
}

void NetCDFReader_Destroy(NetCDFReader *reader) {
    free(reader->xf);
    free(reader->yf);
    free(reader->zf);
    free(reader);
}

NetCDFWriter *NetCDFWriter_Create(const char *filename) {
    NetCDFWriter *writer;

    writer = calloc(1, sizeof(*writer));
    snprintf(writer->filename, NETCDF_MAX_LEN, "%s.nc", filename);

    return writer;
}

void NetCDFWriter_Destroy(NetCDFWriter *writer) {
    free(writer);
}
