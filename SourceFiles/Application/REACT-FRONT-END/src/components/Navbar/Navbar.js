import React from 'react';
import { NavLink } from 'react-router-dom';
import './Navbar.css';

function Navbar() {
  return (
    <nav className="navbar">
      <a href="/" className="site-name nav-item">AlzhiScan.</a>
      <NavLink end to="/" className={({ isActive }) => isActive ? "nav-item active links" : "nav-item links"}>
        Home
      </NavLink>
      <NavLink to="/instructions" className={({ isActive }) => isActive ? "nav-item active links" : "nav-item links"}>
        Instructions
      </NavLink>
      <NavLink to="/predict" className={({ isActive }) => isActive ? "nav-item active links" : "nav-item links"}>
        Predict
      </NavLink>
    </nav>
  );
}

export default Navbar;
