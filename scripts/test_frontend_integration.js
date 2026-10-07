const http = require('http');

async function testBackendHealth() {
  return new Promise((resolve, reject) => {
    http.get('http://54.221.136.83:8000/health', (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => resolve({ status: res.statusCode, body: JSON.parse(data) }));
    }).on('error', reject);
  });
}

async function testBackendCompletion(modelName, promptText) {
  return new Promise((resolve, reject) => {
    const payload = JSON.stringify({
      model: modelName,
      messages: [{ role: "user", content: promptText }],
      temperature: 0.7
    });

    const req = http.request('http://54.221.136.83:8000/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(payload)
      }
    }, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => resolve({ status: res.statusCode, body: JSON.parse(data) }));
    });

    req.on('error', reject);
    req.write(payload);
    req.end();
  });
}

async function runAllTests() {
  console.log("=========================================");
  console.log("RUNNING AUTOMATED GATEWAY INTEGRATION TESTS");
  console.log("=========================================");

  try {
    const health = await testBackendHealth();
    console.log("✓ Health Check Response:", health);

    const modelsToTest = [
      "routemem-auto",
      "gemini-2.5-flash",
      "llama-3.3-70b-versatile",
      "qwen-2.5-coder-32b",
      "deepseek-r1-distill-llama-70b"
    ];

    for (const model of modelsToTest) {
      console.log(`\nTesting Completion Dispatch for Model: [${model}]...`);
      const comp = await testBackendCompletion(model, "Explain QuickSort algorithm in 2 sentences.");
      console.log(`Status: ${comp.status}`);
      console.log(`Model Selected in Backend: ${comp.body.model}`);
      console.log(`Content Output: ${comp.body.choices?.[0]?.message?.content?.substring(0, 100)}...`);
      console.log(`Trace Metadata:`, comp.body.routemem_trace);
    }
    console.log("\nALL INTEGRATION TESTS COMPLETED SUCCESSFULLY.");
  } catch (err) {
    console.error("❌ Integration Test Failed:", err);
  }
}

runAllTests();
