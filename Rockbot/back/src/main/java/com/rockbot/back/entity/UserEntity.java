package com.rockbot.back.entity;

import com.rockbot.back.dto.request.auth.SignUpRequestDto;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
@AllArgsConstructor
@Entity(name = "user")
@Table(name = "user")
public class UserEntity {

    @Id
    @Column(name = "user_id")
    private String userId;

    private String password;
    private String name;
    private String email;
    private String type;
    private String role;

    @Column(name = "department_id") // 부서 ID 필드 추가
    private int departmentId;

    public UserEntity(SignUpRequestDto dto) {
        this.userId = dto.getId();
        this.password = dto.getPassword();
        this.name = dto.getName();
        this.email = dto.getEmail();
        this.type = "app";
        this.role = "ROLE_USER";
        this.departmentId = 1;
    }

    public UserEntity(String userId, String email, String name, String type) {
        this.userId = userId;
        this.password = "passw0rd";
        this.name = name;
        this.email = email;
        this.type = type;
        this.role = "ROLE_USER";
        this.departmentId = 1;
    }

    // OAuth2UserServiceImplement에서 사용하는 새로운 생성자
    public UserEntity(String userId, String email, String type) {
        this.userId = userId;
        this.password = "passw0rd"; // 기본 비밀번호 설정
        this.name = "default_name"; // 기본 이름 설정
        this.email = email;
        this.type = type;
        this.role = "ROLE_USER"; // 기본 역할 설정
        this.departmentId = 1;
    }

    public String getUserId() {
        return userId;
    }

    public String getName() {
        return name;
    }

    public int getDepartmentId() {
        return departmentId;
    }

    public void setDepartmentId(int departmentId) {
        this.departmentId = departmentId;
    }
}
