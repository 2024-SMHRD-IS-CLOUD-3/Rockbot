package com.rockbot.back.dto.request.auth;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class SignUpRequestDto {

    @NotBlank(message = "User ID cannot be blank.")
    private String id;

    @NotBlank(message = "Password cannot be blank.")
    @Pattern(regexp = "^(?=.*[a-zA-Z])(?=.*[0-9])[a-zA-Z0-9]{8,13}$", message = "Password must be 8-13 characters long and contain both letters and numbers.")
    private String password;

    @NotBlank(message = "Name cannot be blank.")
    private String name;

    @Email(message = "Email should be valid.")
    private String email;

    @NotBlank(message = "Certification number cannot be blank.")
    private String certificationNumber;

    @NotNull(message = "Department ID cannot be null.")
    private Integer departmentId = 1;

}
