package com.rockbot.back.entity;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.persistence.Column;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "department") // 실제 테이블 이름과 일치해야 함
public class DepartmentEntity {

    @Id
    @Column(name = "department_id") // `department_id` 컬럼에 매핑
    private int id;

    @Column(name = "department_name")
    private String departmentName;
}
