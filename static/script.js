const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const msg = document.getElementById('msg');

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
  });

function capture() {
  const name = document.getElementById('name').value.trim();
  if (!name) {
    msg.textContent = "Please enter a name";
    msg.style.color = "red";
    return;
  }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  const imageData = canvas.toDataURL('image/jpeg');

  fetch('/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: name, image: imageData })
  })
  .then(res => res.json())
  .then(data => {
    msg.textContent = data.msg;
    msg.style.color = data.color || "green";
  })
  .catch(err => {
    msg.textContent = "Error: " + err;
    msg.style.color = "red";
  });
}
