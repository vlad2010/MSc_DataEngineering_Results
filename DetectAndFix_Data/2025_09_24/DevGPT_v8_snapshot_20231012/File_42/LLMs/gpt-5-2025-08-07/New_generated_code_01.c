#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

/* Security hardening knobs */
#ifndef MAX_BUILD_LOG_BYTES
#define MAX_BUILD_LOG_BYTES (16u * 1024u * 1024u) /* 16 MiB hard cap to avoid OOM (CWE-789) */
#endif

typedef struct BuildCallbackCtx {
    cl_device_id device;
    size_t max_log_bytes;
    int log_to_stderr; /* 0 = suppress logs (production), 1 = print logs (debug) */
} BuildCallbackCtx;

static void secure_free_ctx(BuildCallbackCtx **pctx) {
    if (pctx && *pctx) {
        free(*pctx);
        *pctx = NULL;
    }
}

void CL_CALLBACK buildCallback(cl_program program, void* user_data) {
    BuildCallbackCtx *ctx = (BuildCallbackCtx*)user_data;
    if (!ctx) {
        /* No context; nothing we can safely do */
        return;
    }

    cl_int err = CL_SUCCESS;
    cl_build_status build_status = CL_BUILD_NONE;
    size_t log_size = 0;

    /* Query build status for the specific device passed via user_data (avoid global race - CWE-362) */
    err = clGetProgramBuildInfo(program, ctx->device,
                                CL_PROGRAM_BUILD_STATUS,
                                sizeof(build_status), &build_status, NULL);
    if (err != CL_SUCCESS) {
        if (ctx->log_to_stderr) {
            fprintf(stderr, "Error al obtener el estado de compilación del programa: %d\n", err);
        }
        secure_free_ctx(&ctx);
        return;
    }

    if (build_status != CL_BUILD_ERROR) {
        /* Nothing to print; clean up context and return */
        secure_free_ctx(&ctx);
        return;
    }

    /* Get required log size */
    err = clGetProgramBuildInfo(program, ctx->device,
                                CL_PROGRAM_BUILD_LOG,
                                0, NULL, &log_size);
    if (err != CL_SUCCESS) {
        if (ctx->log_to_stderr) {
            fprintf(stderr, "Error al obtener el tamaño del registro de compilación: %d\n", err);
        }
        secure_free_ctx(&ctx);
        return;
    }

    if (log_size == 0) {
        secure_free_ctx(&ctx);
        return;
    }

    /* Enforce a maximum to prevent OOM/DoS (CWE-789).
       OpenCL requires the buffer to be at least log_size bytes, so if we cap it,
       we cannot retrieve the log. We choose to skip retrieval and warn. */
    if (log_size > ctx->max_log_bytes) {
        if (ctx->log_to_stderr) {
            fprintf(stderr,
                    "El registro de compilación es demasiado grande: %zu bytes (límite %zu). "
                    "Se omite para evitar un consumo excesivo de memoria.\n",
                    log_size, ctx->max_log_bytes);
        }
        secure_free_ctx(&ctx);
        return;
    }

    /* Allocate log_size + 1 to defensively ensure an extra NUL (CWE-170/CWE-131) */
    size_t alloc_size = log_size;
    if (alloc_size < SIZE_MAX) {
        alloc_size += 1; /* safe since we checked above */
    }
    char *build_log = (char*)malloc(alloc_size);
    if (!build_log) {
        if (ctx->log_to_stderr) {
            fprintf(stderr, "Memoria insuficiente al reservar %zu bytes para el registro de compilación.\n", alloc_size);
        }
        secure_free_ctx(&ctx);
        return;
    }
    build_log[alloc_size - 1] = '\0'; /* sentinel */

    /* Retrieve the build log; spec says returned size includes the NUL terminator */
    err = clGetProgramBuildInfo(program, ctx->device,
                                CL_PROGRAM_BUILD_LOG,
                                log_size, build_log, NULL);
    if (err != CL_SUCCESS) {
        if (ctx->log_to_stderr) {
            fprintf(stderr, "Error al obtener el registro de compilación: %d\n", err);
        }
        free(build_log);
        secure_free_ctx(&ctx);
        return;
    }

    /* Defensive NUL-termination; if log_size >= 1, ensure last valid byte is NUL */
    if (log_size >= 1) {
        build_log[log_size - 1] = '\0';
    } else {
        build_log[0] = '\0';
    }

    if (ctx->log_to_stderr) {
        /* Avoid %s trusting unknown NULs; compute bounded length and write */
        size_t text_len = strnlen(build_log, log_size);
        fprintf(stderr, "Registro de compilación (longitud %zu):\n", text_len);
        (void)fwrite(build_log, 1, text_len, stderr);
        fputc('\n', stderr);
        fflush(stderr);
    }

    free(build_log);
    secure_free_ctx(&ctx);
}

/* Helper to build a program safely using the hardened callback.
   The ctx is heap-allocated and owned by the callback, which frees it.
   log_to_stderr = 0 in production to avoid information exposure (CWE-200). */
cl_int safeBuildProgramWithCallback(cl_program program,
                                    cl_device_id device,
                                    const char *options,
                                    int log_to_stderr)
{
    BuildCallbackCtx *ctx = (BuildCallbackCtx*)calloc(1, sizeof(BuildCallbackCtx));
    if (!ctx) {
        return CL_OUT_OF_HOST_MEMORY;
    }
    ctx->device = device;
    ctx->max_log_bytes = MAX_BUILD_LOG_BYTES;
    ctx->log_to_stderr = log_to_stderr ? 1 : 0;

    /* The callback will free ctx when invoked. If clBuildProgram fails immediately,
       we must free ctx here. */
    cl_int err = clBuildProgram(program, 1, &device, options, buildCallback, ctx);
    if (err != CL_SUCCESS) {
        secure_free_ctx(&ctx);
    }
    return err;
}