package com.rockbot.back.provider;

import java.security.Key;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Date;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import com.nimbusds.jose.util.StandardCharset;

import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.SignatureAlgorithm;
import io.jsonwebtoken.security.Keys;

@Component
public class JwtProvider {

    @Value("${secret-key}")
    private String secretKey;

    public String create(String userId, String name, int departmentId) {

        Date expiredDate = Date.from(Instant.now().plus(1, ChronoUnit.HOURS));
        Key key = Keys.hmacShaKeyFor(secretKey.getBytes(StandardCharset.UTF_8));

        String jwt = Jwts.builder()
                .signWith(key, SignatureAlgorithm.HS256)
                .setSubject(userId).claim("name", name).claim("departmentId", 1).setIssuedAt(new Date())
                .setExpiration(expiredDate)
                .compact();

        return jwt;

    }

    public String validate(String jwt) {

        String subject = null;
        Key key = Keys.hmacShaKeyFor(secretKey.getBytes(StandardCharset.UTF_8));

        try {

            subject = Jwts.parserBuilder()
                    .setSigningKey(key)
                    .build()
                    .parseClaimsJws(jwt)
                    .getBody()
                    .getSubject();

        } catch (Exception exception) {
            exception.printStackTrace();
            return null;
        }

        return subject;

    }

}
