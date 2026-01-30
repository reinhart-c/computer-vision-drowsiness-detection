const ws = new WebSocket("ws://localhost:8000/ws");


const img = document.getElementById("videoStream");
const statusBox = document.getElementById("status");


const metricEls = [
    document.getElementById("m1"),
    document.getElementById("m2"),
    document.getElementById("m3"),
    document.getElementById("m4"),
    document.getElementById("m5"),
];


ws.onmessage = (event) => {
    const data = JSON.parse(event.data);


    if (data.frame !== null) {
        img.src = "data:image/jpeg;base64," + data.frame;
    }


    if (data.features !== null) {
        for (let i = 0; i < 5; i++) {
            metricEls[i].textContent = data.features[i].toFixed(2);
        }
    }


    if (data.prediction !== null) {
        pred = data.prediction
        console.log(pred)
        if (pred == -1){
            statusBox.textContent = "Loading";
        }else if (pred == 0) {
            statusBox.textContent = "Alert";
        }else if (pred == 1) {
            statusBox.textContent = "Low Vigilance";
        }else if (pred == 2) {
            statusBox.textContent = "Drowsy";
        }
    }
};


ws.onerror = () => {
    statusBox.textContent = "WebSocket error";
};