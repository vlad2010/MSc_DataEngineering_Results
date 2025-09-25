#include <stdio.h>
#include <arpa/inet.h>

int main() {
    uint32_t value = 0x03da0001; // 예시로 16진수 데이터를 정의

    // 데이터를 빅 엔디안으로 변환
    uint32_t bigEndianValue = htonl(value);

    // 변환된 빅 엔디안 값을 출력
    printf("빅 엔디안 값: 0x%08x\n", bigEndianValue);

    return 0;
}
