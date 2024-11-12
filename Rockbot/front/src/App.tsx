import React, { ChangeEvent, useState } from 'react';
import './App.css'; 
import InputBox from 'components/InputBox';
import { Route, Routes } from 'react-router-dom';
import SignUp from 'views/Authentication/SignUp';
import SignIn from 'views/Authentication/SignIn';
import OAuth from 'views/Authentication/OAuth';
import ChatBot from 'views/Authentication/CahtBot/ChatBot';
import Admin from 'views/Authentication/AdminPage/AdminPage';

function App() {

  return (
    <Routes>
      <Route path='/auth'>
        <Route path='sign-up' element={<SignUp />} />
        <Route path='sign-in' element={<SignIn />} />
        <Route path='chat' element={<ChatBot />} />
        <Route path='admin' element={<Admin />} />
        <Route path='oauth-response/:token/:expirationTime' element={<OAuth />} />
        
        </Route>
    </Routes>
  );
}

export default App;
