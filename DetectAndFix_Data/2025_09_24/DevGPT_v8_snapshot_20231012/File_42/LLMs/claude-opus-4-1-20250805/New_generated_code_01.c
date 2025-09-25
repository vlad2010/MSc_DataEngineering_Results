#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

#define MAX_BUILD_LOG_SIZE (1024 * 1024) // 1MB max log size

// Assuming device_id is properly defined elsewhere
extern cl_device_id device_id;

void CL_CALLBACK buildCallback(cl_program program, void* user_data) {
    cl_int errcode;
    cl_build_status build_status;
    size_t build_log_size;
    char* build_log = NULL;

    // Validate input parameters
    if (program == NULL) {
        printf("Error: programa NULL recibido en callback\n");
        return;
    }

    // Obtiene el estado de compilación del programa
    errcode = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_STATUS, 
                                   sizeof(cl_build_status), &build_status, NULL);
    if (errcode != CL_SUCCESS) {
        printf("Error al obtener el estado de compilación del programa: %d\n", errcode);
        return;
    }

    // Si el estado de compilación es CL_BUILD_ERROR, obtiene el registro de compilación
    if (build_status == CL_BUILD_ERROR) {
        errcode = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 
                                       0, NULL, &build_log_size);
        if (errcode != CL_SUCCESS) {
            printf("Error al obtener el tamaño del registro de compilación: %d\n", errcode);
            return;
        }

        // Validate build log size to prevent excessive memory allocation
        if (build_log_size == 0) {
            printf("Registro de compilación vacío\n");
            return;
        }
        
        if (build_log_size > MAX_BUILD_LOG_SIZE) {
            printf("Advertencia: registro de compilación demasiado grande (%zu bytes), truncando a %d bytes\n", 
                   build_log_size, MAX_BUILD_LOG_SIZE);
            build_log_size = MAX_BUILD_LOG_SIZE;
        }

        // Allocate memory with null check
        build_log = (char*)calloc(build_log_size + 1, sizeof(char)); // +1 for null terminator
        if (build_log == NULL) {
            printf("Error: no se pudo asignar memoria para el registro de compilación (%zu bytes)\n", 
                   build_log_size);
            return;
        }

        errcode = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 
                                       build_log_size, build_log, NULL);
        if (errcode != CL_SUCCESS) {
            printf("Error al obtener el registro de compilación: %d\n", errcode);
            free(build_log);
            build_log = NULL;
            return;
        }

        // Ensure null termination
        build_log[build_log_size] = '\0';

        // Print build log safely - avoid format string vulnerability
        printf("Registro de compilación:\n");
        printf("%s", build_log);
        if (build_log[strlen(build_log) - 1] != '\n') {
            printf("\n");
        }

        // Clean up
        free(build_log);
        build_log = NULL;
    } else if (build_status == CL_BUILD_SUCCESS) {
        printf("Compilación exitosa\n");
    } else if (build_status == CL_BUILD_IN_PROGRESS) {
        printf("Compilación en progreso\n");
    }
}