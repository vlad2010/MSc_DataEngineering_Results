void CL_CALLBACK buildCallback(cl_program program, void* user_data) {
    cl_int errcode;
    cl_build_status build_status;
    size_t build_log_size;

    // Obtiene el estado de compilación del programa
    errcode = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL);
    if (errcode != CL_SUCCESS) {
        printf("Error al obtener el estado de compilación del programa: %d\n", errcode);
        return;
    }

    // Si el estado de compilación es CL_BUILD_ERROR, obtiene el registro de compilación
    if (build_status == CL_BUILD_ERROR) {
        errcode = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_size);
        if (errcode != CL_SUCCESS) {
            printf("Error al obtener el tamaño del registro de compilación: %d\n", errcode);
            return;
        }

        char* build_log = (char*)malloc(build_log_size);
        errcode = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, build_log_size, build_log, NULL);
        if (errcode != CL_SUCCESS) {
            printf("Error al obtener el registro de compilación: %d\n", errcode);
            free(build_log);
            return;
        }

        // Imprime el registro de compilación
        printf("Registro de compilación:\n%s\n", build_log);

        free(build_log);
    }
}
