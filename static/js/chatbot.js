document.addEventListener('DOMContentLoaded', function() {
    var socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port);

    var form = document.getElementById('chat-form');
    var chatBox = document.getElementById('chat-box');

    form.addEventListener('submit', function(event) {
        event.preventDefault();
        var userInput = document.getElementById('user-input');
        var message = userInput.value.trim();
        if (message) {
            addMessage('user', message);
            socket.emit('message', message);
            userInput.value = '';
        }
    });

    socket.on('response', function(data) {
        addMessage('bot', data.response);
    });

    function addMessage(sender, message) {
        var messageDiv = document.createElement('div');
        messageDiv.classList.add('chat-message', sender + '-message');
        messageDiv.innerHTML = message;
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
});
