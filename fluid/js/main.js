import EventBus from "./utils/EventBus.js";
window.EventBus = EventBus;
import WebGL from "./modules/WebGL.js";

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

    const progressBar = document.getElementById('progress-bar');
    const loadingScreen = document.getElementById('loading');
    const readyText = document.getElementById('ready-text');

    function updateTextAfter(seconds, text) {
        setTimeout(() => {
            readyText.textContent = text;
        }, seconds * 1000);
    }

    updateTextAfter(1, "Loading...");
    updateTextAfter(2, "20% Complete");
    updateTextAfter(3, "24% Complete");
    updateTextAfter(4, "31% Complete");
    updateTextAfter(5, "39% Complete");
    updateTextAfter(6, "48% Complete");
    updateTextAfter(7, "62% Complete");
    updateTextAfter(8, "82% Complete");
    updateTextAfter(9, "99% Complete");
    updateTextAfter(10, "100% Complete");

    readyText.textContent = 'Loading...';

    progressBar.style.width = '100%';

    progressBar.addEventListener('transitionend', () => {
        readyText.textContent = "Loading...";
        readyText.style.animation = 'none';

        setTimeout(() => {
            loadingScreen.style.opacity = '0';

            setTimeout(() => {
                loadingScreen.style.display = 'none';
            }, 500);
        }, 5000);
    });

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
    let card4Position = 250;
    let card5Position = 290;
    let card6Position = 330;
    let card7Position = 370;
    let card8Position = 410;
    let card9Position = 450;
    let card10Position = 490;
    let card11Position = 530;
    let card12Position = 570;
    let card13Position = 610;

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
            const card4 = document.getElementById('card4');
            const card5 = document.getElementById('card5');
            const card6 = document.getElementById('card6');
            const card7 = document.getElementById('card7');
            const card8 = document.getElementById('card8');
            const card9 = document.getElementById('card9');
            const card10 = document.getElementById('card10');
            const card11 = document.getElementById('card11');
            const card12 = document.getElementById('card12');
            const card13 = document.getElementById('card13');

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
                card4.classList.add('visible');
                card.style.right = '10%';
                cardVisible = true;
                card5.classList.add('visible');
                card.style.right = '10%';
                cardVisible = true;
                card6.classList.add('visible');
                card.style.right = '10%';
                cardVisible = true;
                card7.classList.add('visible');
                card.style.right = '10%';
                cardVisible = true;
                card8.classList.add('visible');
                card.style.right = '10%';
                cardVisible = true;
                card9.classList.add('visible');
                card.style.right = '10%';
                cardVisible = true;
                card10.classList.add('visible');
                card.style.right = '10%';
                cardVisible = true;
                card11.classList.add('visible');
                card.style.right = '10%';
                cardVisible = true;
                card12.classList.add('visible');
                card.style.right = '10%';
                cardVisible = true;
                card13.classList.add('visible');
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
                card4.style.right = '-100%'
                cardVisible = false
                card5.style.right = '-100%'
                cardVisible = false
                card6.style.right = '-100%'
                cardVisible = false
                card7.style.right = '-100%'
                cardVisible = false
                card8.style.right = '-100%'
                cardVisible = false
                card9.style.right = '-100%'
                cardVisible = false
                card10.style.right = '-100%'
                cardVisible = false
                card11.style.right = '-100%'
                cardVisible = false
                card12.style.right = '-100%'
                cardVisible = false
                card13.style.right = '-100%'
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
            document.getElementById('card4').classList.remove('visible');
            isCardMovingRight = false;
            document.getElementById('card5').classList.remove('visible');
            isCardMovingRight = false;
            document.getElementById('card6').classList.remove('visible');
            isCardMovingRight = false;
            document.getElementById('card7').classList.remove('visible');
            isCardMovingRight = false;
            document.getElementById('card8').classList.remove('visible');
            isCardMovingRight = false;
            document.getElementById('card9').classList.remove('visible');
            isCardMovingRight = false;
            document.getElementById('card10').classList.remove('visible');
            isCardMovingRight = false;
            document.getElementById('card11').classList.remove('visible');
            isCardMovingRight = false;
            document.getElementById('card12').classList.remove('visible');
            isCardMovingRight = false;
            document.getElementById('card13').classList.remove('visible');
            isCardMovingRight = false;
        }

        let scrollDirection = scrolled > lastScrollTop ? 'down' : 'up';
        lastScrollTop = scrolled <= 0 ? 0 : scrolled;

        if (scrolled >= cardAppearThreshold) {
            if (scrollDirection === 'down') {
                cardPosition = Math.max(cardPosition - 7, -4000);
                card1Position = cardPosition + 130;
                card2Position = card1Position + 130;
                card3Position = card2Position + 130;
                card4Position = card3Position + 130;
                card5Position = card4Position + 130;
                card6Position = card5Position + 130;
                card7Position = card6Position + 130;
                card8Position = card7Position + 130;
                card9Position = card8Position + 130;
                card10Position = card9Position + 130;
                card11Position = card10Position + 130;
                card12Position = card11Position + 130;
                card13Position = card12Position + 130;
            } else {
                cardPosition = Math.min(cardPosition + 7, 4000);
                card1Position = cardPosition + 130;
                card2Position = card1Position + 130;
                card3Position = card2Position + 130;
                card4Position = card3Position + 130;
                card5Position = card4Position + 130;
                card6Position = card5Position + 130;
                card7Position = card6Position + 130;
                card8Position = card7Position + 130;
                card9Position = card8Position + 130;
                card10Position = card9Position + 130;
                card11Position = card10Position + 130;
                card12Position = card11Position + 130;
                card13Position = card12Position + 130;
            }
            updateCardPosition(cardPosition, card1Position, card2Position, card3Position, card4Position, card5Position, card6Position, card7Position, card8Position, card9Position, card10Position, card11Position, card12Position, card13Position);
        }

        let opacity = 1 - (scrolled - startThreshold) / (endThreshold - startThreshold);
        document.getElementById('about').style.opacity = opacity;
        document.getElementById('portfolio').style.opacity = opacity;
    });

    function updateCardPosition(pos, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9, pos10, pos11, pos12, pos13) {
        const card = document.getElementById('card');
        const card1 = document.getElementById('card1');
        const card2 = document.getElementById('card2');
        const card3 = document.getElementById('card3');
        const card4 = document.getElementById('card4');
        const card5 = document.getElementById('card5');
        const card6 = document.getElementById('card6');
        const card7 = document.getElementById('card7');
        const card8 = document.getElementById('card8');
        const card9 = document.getElementById('card9');
        const card10 = document.getElementById('card10');
        const card11 = document.getElementById('card11');
        const card12 = document.getElementById('card12');
        const card13 = document.getElementById('card13');
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

        if (!card4) return;
        card4.style.right = `${pos4}%`;
        card4.classList.add('visible');

        if (!card5) return;
        card5.style.right = `${pos5}%`;
        card5.classList.add('visible');

        if (!card6) return;
        card6.style.right = `${pos6}%`;
        card6.classList.add('visible');

        if (!card7) return;
        card7.style.right = `${pos7}%`;
        card7.classList.add('visible');

        if (!card8) return;
        card8.style.right = `${pos8}%`;
        card8.classList.add('visible');

        if (!card9) return;
        card9.style.right = `${pos9}%`;
        card9.classList.add('visible');

        if (!card10) return;
        card10.style.right = `${pos10}%`;
        card10.classList.add('visible');

        if (!card11) return;
        card11.style.right = `${pos11}%`;
        card11.classList.add('visible');

        if (!card12) return;
        card12.style.right = `${pos12}%`;
        card12.classList.add('visible');

        if (!card13) return;
        card13.style.right = `${pos13}%`;
        card13.classList.add('visible');
    }
});