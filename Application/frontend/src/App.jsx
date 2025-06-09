import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import Form from './components/Form'
import Footer from './components/Footer'


function App() {

  return (
    <>
      <h1 align="center">SAR Image Colorization</h1>
      <Form/>
      <Footer/>
    </>
  )
}

export default App
