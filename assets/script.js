function logChoose(n, k) {
  if (k < 0 || k > n) return Number.NEGATIVE_INFINITY;
  const m = Math.min(k, n - k);
  let result = 0;
  for (let i = 1; i <= m; i += 1) result += Math.log(n - m + i) - Math.log(i);
  return result;
}

function conditionalTable(total, rowHigh, columnHigh, oddsRatio) {
  const lower = Math.max(0, rowHigh + columnHigh - total);
  const upper = Math.min(rowHigh, columnHigh);
  const theta = Math.log(oddsRatio);
  const support = [];
  let maximum = Number.NEGATIVE_INFINITY;
  for (let count = lower; count <= upper; count += 1) {
    const logWeight = logChoose(rowHigh, count) + logChoose(total - rowHigh, columnHigh - count) + theta * count;
    support.push([count, logWeight]);
    maximum = Math.max(maximum, logWeight);
  }
  let normalizer = 0;
  let weightedSum = 0;
  support.forEach(([count, logWeight]) => {
    const weight = Math.exp(logWeight - maximum);
    normalizer += weight;
    weightedSum += count * weight;
  });
  const highHigh = weightedSum / normalizer;
  return { lowLow: total - rowHigh - columnHigh + highHigh, lowHigh: columnHigh - highHigh, highLow: rowHigh - highHigh, highHigh };
}

function updateTable() {
  const rowPercent = Number(document.querySelector("#rna-margin").value);
  const columnPercent = Number(document.querySelector("#protein-margin").value);
  const oddsRatio = Math.exp(Number(document.querySelector("#odds-ratio").value) / 100);
  const total = 512;
  const rowHigh = Math.round(total * rowPercent / 100);
  const columnHigh = Math.round(total * columnPercent / 100);
  const table = conditionalTable(total, rowHigh, columnHigh, oddsRatio);
  const cells = { "cell-00": table.lowLow, "cell-01": table.lowHigh, "cell-10": table.highLow, "cell-11": table.highHigh };
  Object.entries(cells).forEach(([id, value]) => {
    const element = document.querySelector(`#${id}`);
    element.textContent = value.toFixed(1);
    element.parentElement.style.setProperty("--fill", String(0.18 + 0.82 * value / total));
  });
  document.querySelector("#rna-output").textContent = `${rowPercent}%`;
  document.querySelector("#protein-output").textContent = `${columnPercent}%`;
  document.querySelector("#odds-output").textContent = oddsRatio.toFixed(2);
  document.querySelector("#rna-check").textContent = `${(100 * rowHigh / total).toFixed(1)}%`;
  document.querySelector("#protein-check").textContent = `${(100 * columnHigh / total).toFixed(1)}%`;
}

document.querySelectorAll("#margin-lab input").forEach((input) => input.addEventListener("input", updateTable));
updateTable();
