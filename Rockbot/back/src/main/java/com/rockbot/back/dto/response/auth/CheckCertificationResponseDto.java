package com.rockbot.back.dto.response.auth;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;

import com.rockbot.back.common.ResponseCode;
import com.rockbot.back.common.ResponseMessage;
import com.rockbot.back.dto.response.ResponseDto;

import lombok.Getter;

@Getter
public class CheckCertificationResponseDto extends ResponseDto {

    private CheckCertificationResponseDto() {
        super();
    }

    public static ResponseEntity<CheckCertificationResponseDto> success() {
        CheckCertificationResponseDto responseBody = new CheckCertificationResponseDto();
        return ResponseEntity.status(HttpStatus.OK).body(responseBody);
    }

    public static ResponseEntity<ResponseDto> certificationFail() {
        ResponseDto responseBody = new ResponseDto(ResponseCode.CERTIFICATION_FAIL, ResponseMessage.CERTIFICATION_FAIL);
        return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(responseBody);
    }

}
