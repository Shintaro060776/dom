import EventBus from "./utils/EventBus";
window.EventBus = EventBus;
import WebGL from "./modules/WebGL";

if (!window.isDev) window.isDev = false;

const webglMng = new WebGL({
    $wrapper: document.body
});

window.scrollToTop = function () {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
};


document.addEventListener('DOMContentLoaded', () => {
    const container = document.createElement('div');
    container.className = 'scroll-down-container';
    document.body.appendChild(container);

    const text = 'Scroll Down';
    const arrow = document.createElement('div');
    arrow.className = 'scroll-down-arrow';
    arrow.textContent = '↓';
    container.appendChild(arrow);

    function animateText() {
        container.querySelectorAll('span').forEach(span => span.remove());

        text.split('').forEach((char, index) => {
            const span = document.createElement('span');
            span.textContent = char;
            span.style.animationDelay = `${0.5 + 0.1 * index}s`;
            container.insertBefore(span, arrow);
        });

        arrow.style.animation = 'none';
        setTimeout(() => {
            arrow.style.animation = '';
        }, 100);
    }

    animateText();

    setInterval(animateText, 3500);

    let isMouseOver = false;

    container.addEventListener('mouseenter', () => {
        isMouseOver = true;
    });

    container.addEventListener('mouseleave', () => {
        isMouseOver = false;
        container.style.transform = `translate(-50%)`;
    });

    document.addEventListener('mousemove', (event) => {
        if (!isMouseOver) return;

        const container = document.querySelector('.scroll-down-container');
        if (!container) return;

        const maxMoveRadius = 50;

        const containerRect = container.getBoundingClientRect();
        const containerCenterX = containerRect.left + containerRect.width / 2;
        const containerCenterY = containerRect.top + containerRect.height / 2;

        let distanceX = event.clientX - containerCenterX;
        let distanceY = event.clientY - containerCenterY;

        const distance = Math.sqrt(distanceX * distanceX + distanceY * distanceY);
        if (distance > maxMoveRadius) {
            const angle = Math.atan2(distanceY, distanceX);
            distanceX = Math.cos(angle) * maxMoveRadius;
            distanceY = Math.sin(angle) * maxMoveRadius;
        }

        container.style.transform = `translate(${distanceX}px, ${distanceY}px)`;

    });

    function generateRandomString(length) {
        const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
        let result = '';
        for (let i = 0; i < length; i++) {
            result += characters.charAt(Math.floor(Math.random() * characters.length));
        }
        return result;
    }

    function changeTextWithRandomStrings(elementId, finalText, totalDuration, intervalDuration) {
        const interval = setInterval(() => {
            document.getElementById(elementId).textContent = generateRandomString(30);
        }, intervalDuration);

        setTimeout(() => {
            clearInterval(interval);
            document.getElementById(elementId).textContent = finalText;
        }, totalDuration);
    }

    const startThreshold = window.innerHeight * 3;
    const switchTextThreshold = window.innerHeight * 6;
    const endThreshold = window.innerHeight * 10;
    const moreTextThreshold = endThreshold + (window.innerHeight * 2);
    const moreStartThreshold = window.innerHeight * 8;
    const moreEndThreshold = window.innerHeight * 10;

    const cardAppearThreshold = moreTextThreshold + (window.innerHeight * 2);
    let textChanged = false;
    let textReset = true;
    let moreTextVisible = false;
    let cardVisible = false;

    let lastScrollTop = 0;
    let isCardMovingRight = false;

    let cardPosition = 90;
    let card1Position = 130;
    let card2Position = 170;
    let card3Position = 210;

    window.addEventListener('scroll', () => {
        const scrolled = window.scrollY;

        if (scrolled >= startThreshold && scrolled <= endThreshold) {
            document.getElementById('about').classList.add('visible');
            document.getElementById('portfolio').classList.add('visible');

            if (scrolled > switchTextThreshold && !textChanged) {
                textChanged = true;
                textReset = false;
                changeTextWithRandomStrings('portfolio', "Welcome to my world of creativity!", 5000, 10);
            }
        } else {
            document.getElementById('about').classList.remove('visible');
            document.getElementById('portfolio').classList.remove('visible');

            if (textReset) {
                textChanged = false;
                document.getElementById('portfolio').textContent = "This is my portfolio site. Please explore and enjoy a lot";
            }
        }

        if (scrolled >= moreTextThreshold && scrolled <= cardAppearThreshold) {
            document.getElementById('more').classList.add('visible');
            document.getElementById('more').classList.remove('move-left');
        } else {
            document.getElementById('more').classList.remove('visible');
            if (scrolled > cardAppearThreshold) {
                document.getElementById('more').classList.add('move-left');
            }
        }

        if (scrolled >= cardAppearThreshold) {
            const card = document.getElementById('card');
            const card1 = document.getElementById('card1');
            const card2 = document.getElementById('card2');
            const card3 = document.getElementById('card3');

            if (!cardVisible && scrollDirection === 'down') {
                card.classList.add('visible');
                card.style.right = '10%';
                cardVisible = true;
                card1.classList.add('visible');
                card.style.right = '10%';
                cardVisible = true;
                card2.classList.add('visible');
                card.style.right = '10%';
                cardVisible = true;
                card3.classList.add('visible');
                card.style.right = '10%';
                cardVisible = true;
            } else if (cardVisible && scrollDirection === 'up') {
                card.style.right = '-100%'
                cardVisible = false
                card1.style.right = '-100%'
                cardVisible = false
                card2.style.right = '-100%'
                cardVisible = false
                card3.style.right = '-100%'
                cardVisible = false
            }
        } else {
            document.getElementById('card').classList.remove('visible');
            isCardMovingRight = false;
            document.getElementById('card1').classList.remove('visible');
            isCardMovingRight = false;
            document.getElementById('card2').classList.remove('visible');
            isCardMovingRight = false;
            document.getElementById('card3').classList.remove('visible');
            isCardMovingRight = false;
        }

        let scrollDirection = scrolled > lastScrollTop ? 'down' : 'up';
        lastScrollTop = scrolled <= 0 ? 0 : scrolled;

        if (scrolled >= cardAppearThreshold) {
            if (scrollDirection === 'down') {
                cardPosition = Math.max(cardPosition - 3, -500);
                card1Position = cardPosition + 130;
                card2Position = card1Position + 130;
                card3Position = card2Position + 130;
            } else {
                cardPosition = Math.min(cardPosition + 3, 500);
                card1Position = cardPosition + 130;
                card2Position = card1Position + 130;
                card3Position = card2Position + 130;
            }
            updateCardPosition(cardPosition, card1Position, card2Position, card3Position);
        }

        let opacity = 1 - (scrolled - startThreshold) / (endThreshold - startThreshold);
        document.getElementById('about').style.opacity = opacity;
        document.getElementById('portfolio').style.opacity = opacity;
    });

    function updateCardPosition(pos, pos1, pos2, pos3) {
        const card = document.getElementById('card');
        const card1 = document.getElementById('card1');
        const card2 = document.getElementById('card2');
        const card3 = document.getElementById('card3');
        if (!card) return;
        card.style.right = `${pos}%`;
        card.classList.add('visible');

        if (!card1) return;
        card1.style.right = `${pos1}%`;
        card1.classList.add('visible');

        if (!card2) return;
        card2.style.right = `${pos2}%`;
        card2.classList.add('visible');

        if (!card3) return;
        card3.style.right = `${pos3}%`;
        card3.classList.add('visible');
    }
});