package com.rockbot.back.entity;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.persistence.Column;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDate;
import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "travel_request")
public class TravelEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "request_id")
    private Long requestId;

    @Column(name = "user_id")
    private String userId;

    @Column(name = "department_id")
    private int departmentId;

    private String destination;

    @Column(name = "travel_date")
    private LocalDate travelDate;

    @Column(name = "return_date")
    private LocalDate returnDate;

    private String reason;

    private String status;

    @Column(name = "submission_date")
    private LocalDateTime submissionDate;

    private String name;
}
