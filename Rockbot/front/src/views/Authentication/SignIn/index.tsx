import InputBox from 'components/InputBox';
import React, { useState, useRef, ChangeEvent, KeyboardEvent, useEffect } from 'react';
import './style.css';
import { useNavigate } from 'react-router-dom';
import { SignInRequestDto } from 'apis/request/auth';
import { SignInRequest, SNS_SIGN_IN_URL } from 'apis';
import { ResponseBody } from 'types';
import { SignInResponseDto } from 'apis/response/auth';
import { ResponseCode } from 'types/enums';
import { useCookies } from 'react-cookie';

export default function SignIn() {

    const idRef = useRef<HTMLInputElement | null>(null);
    const passwordRef = useRef<HTMLInputElement | null>(null);

    const [cookie, setCookie] = useCookies();

    const [id, setId] = useState<string>('');
    const [password, setPassword] = useState<string>('');

    const [message, setMessage] = useState<string>('');
    const [loggedName, setLoggedName] = useState<string | null>(null);

    const navigate = useNavigate();

    const signInButtonClass = id && password ? 'primary-button-lg' : 'disable-button-lg';

    const signInResponse = (ResponseBody: ResponseBody<SignInResponseDto>) => {
        if (!ResponseBody) return;
    
        const { code } = ResponseBody;
        if (code === ResponseCode.VALIDATION_FAIL) {
            alert('아이디와 비밀번호를 입력하세요.');
        }
        if (code === ResponseCode.SIGN_IN_FAIL) {
            setMessage('로그인 정보가 일치하지 않습니다.');
        }
        if (code === ResponseCode.DATABASE_ERROR) {
            alert('데이터베이스 오류입니다.');
        }
        if (code !== ResponseCode.SUCCESS) {
            return;
        }
    
        const { token, expirationTime, userId, name, departmentId } = ResponseBody as SignInResponseDto;
    
        const now = new Date().getTime();
        const expires = new Date(now + expirationTime * 1000);
    
        // accessToken과 userId를 쿠키에 저장
        setCookie('accessToken', token, { expires, path: '/' });
        setCookie('userId', userId, { expires, path: '/' });
    
        // userInfo는 localStorage에 저장
        localStorage.setItem('userInfo', JSON.stringify({ userId, token, name, departmentId }));
    
        setLoggedName(name);
    
        // 로그인 후 조건에 따라 페이지 이동
        if (userId === 'Admin') {
            // Admin 사용자인 경우 관리 페이지로 이동
            navigate('/auth/admin');
        } else {
            // 일반 사용자일 경우 채팅창으로 이동
            navigate('/auth/chat', { state: { name, departmentId } });
        }
    };    

    useEffect(() => {
        const storedUserInfo = localStorage.getItem('userInfo');
        if (storedUserInfo) {
            const parsedUserInfo = JSON.parse(storedUserInfo);
            setLoggedName(parsedUserInfo.name);
        }
    }, []);

    const onIdChangeHandler = (event: ChangeEvent<HTMLInputElement>) => {
        const { value } = event.target;
        setId(value);
        setMessage('');
    };

    const onPasswordChangeHandler = (event: ChangeEvent<HTMLInputElement>) => {
        const { value } = event.target;
        setPassword(value);
        setMessage('');
    };

    const onSignUpButtonClickHandler = () => {
        navigate('/auth/sign-up');
    };

    const onSignInButtonClickHandler = () => {
        if (!id || !password) {
            alert('아이디와 비밀번호 모두 입력하세요.');
            return;
        }
        
        const requestBody: SignInRequestDto = { id, password };
        SignInRequest(requestBody).then(signInResponse);
    };

    const onSnsSignInButtonClickHandler = (type: 'kakao' | 'naver') => {
        window.location.href = SNS_SIGN_IN_URL(type);
    };

    const onIdKeyDownHandler = (event: KeyboardEvent<HTMLInputElement>) => {
        if (event.key !== 'Enter') return;
        if (!passwordRef.current) return;
        passwordRef.current.focus();
    };

    const onPasswordKeyDownHandler = (event: KeyboardEvent<HTMLInputElement>) => {
        if (event.key !== 'Enter') return;
        onSignInButtonClickHandler();
    };

    return (
        <div id='sign-in-wrapper'>
            <div className='sign-in-image'></div>
            <div className='sign-in-container'>
                <div className='sign-in-box'>
                    <div className='sign-in-title'></div>
                    <div className='sign-in-content-box'>
                    <div className='sign-in-content-input-box'>
                        <InputBox ref={idRef} title='아이디' placeholder='아이디를 입력해주세요' type='text' value={id} onChange={onIdChangeHandler} onKeyDown={onIdKeyDownHandler}></InputBox>
                        <InputBox ref={passwordRef} title='비밀번호' placeholder='비밀번호를 입력해주세요' type='password' value={password} isErrorMessage message={message} onChange={onPasswordChangeHandler} onKeyDown={onPasswordKeyDownHandler}></InputBox>
                    </div>

                    <div className='sign-in-content-button-box'>
                        <div className={`${signInButtonClass} full-width`} onClick={onSignInButtonClickHandler}>{'로그인'}</div>
                        <div className='text-link-lg full-width' onClick={onSignUpButtonClickHandler}>{'회원가입'}</div>
                        <div className='sign-in-content-divider'></div>
                        <div className='sign-in-content-sns-sign-in-box'>
                            <div className='sign-in-content-sns-sign-in-title'>{'SNS 로그인'}</div>
                            <div className='sign-in-content-sns-sign-in-button-box'>
                                <div className='kakao-sign-in-button' onClick={() => onSnsSignInButtonClickHandler('kakao')}></div>
                                <div className='naver-sign-in-button'  onClick={() => onSnsSignInButtonClickHandler('naver')}></div>
                            </div>
                        </div>
                    </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
