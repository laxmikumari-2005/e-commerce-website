let menu = document.querySelector('#menu-btn');
let navbar = document.querySelector('.navbar');

menu.onclick = () =>{
    menu.classList.toggle('fa-times');
    navbar.classlist.toggle('active');
}

window.onscroll = () =>{
    menu.classlist.remove('fa-times');
    navbar.classlist.remove('active');
}