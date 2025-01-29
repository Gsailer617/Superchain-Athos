class TaxManager {
    constructor() {
        this.transactions = [];
        this.initializeEventListeners();
        this.updateInterval = setInterval(() => this.fetchTaxData(), 60000); // Update every minute
    }

    initializeEventListeners() {
        document.getElementById('download-csv').addEventListener('click', () => this.downloadCSV());
        document.getElementById('generate-8949').addEventListener('click', () => this.generateForm8949());
    }

    async fetchTaxData() {
        try {
            const response = await fetch('/api/dashboard/tax-data');
            const data = await response.json();
            this.transactions = data.transactions;
            this.updateTaxDisplay();
        } catch (error) {
            console.error('Failed to fetch tax data:', error);
        }
    }

    updateTaxDisplay() {
        this.updateTaxSummary();
        this.updateTransactionsTable();
    }

    updateTaxSummary() {
        const summary = this.calculateTaxSummary();
        document.getElementById('total-gains').textContent = summary.totalGains.toFixed(2);
        document.getElementById('short-term-gains').textContent = summary.shortTermGains.toFixed(2);
        document.getElementById('long-term-gains').textContent = summary.longTermGains.toFixed(2);
    }

    calculateTaxSummary() {
        const summary = {
            totalGains: 0,
            shortTermGains: 0,
            longTermGains: 0
        };

        this.transactions.forEach(tx => {
            const gain = tx.proceeds - tx.costBasis;
            summary.totalGains += gain;
            
            if (this.isLongTerm(tx.acquiredDate, tx.soldDate)) {
                summary.longTermGains += gain;
            } else {
                summary.shortTermGains += gain;
            }
        });

        return summary;
    }

    updateTransactionsTable() {
        const tbody = document.getElementById('tax-transactions-body');
        tbody.innerHTML = '';

        this.transactions
            .sort((a, b) => b.soldDate - a.soldDate) // Most recent first
            .forEach(tx => {
                const row = document.createElement('tr');
                const gain = tx.proceeds - tx.costBasis;
                const holdingPeriod = this.calculateHoldingPeriod(tx.acquiredDate, tx.soldDate);
                
                row.innerHTML = `
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        ${new Date(tx.soldDate).toLocaleDateString()}
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        ${tx.type}
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        ${tx.asset}
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        ${tx.quantity.toFixed(6)}
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        $${tx.costBasis.toFixed(2)}
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        $${tx.proceeds.toFixed(2)}
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm ${gain >= 0 ? 'text-green-600' : 'text-red-600'}">
                        ${gain >= 0 ? '+' : ''}$${gain.toFixed(2)}
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        ${holdingPeriod}
                    </td>
                `;
                tbody.appendChild(row);
            });
    }

    isLongTerm(acquiredDate, soldDate) {
        const holdingPeriod = soldDate - acquiredDate;
        const oneYear = 365 * 24 * 60 * 60 * 1000;
        return holdingPeriod >= oneYear;
    }

    calculateHoldingPeriod(acquiredDate, soldDate) {
        const days = Math.floor((soldDate - acquiredDate) / (24 * 60 * 60 * 1000));
        if (days >= 365) {
            const years = Math.floor(days / 365);
            const remainingDays = days % 365;
            return `${years}y ${remainingDays}d`;
        }
        return `${days}d`;
    }

    async downloadCSV() {
        const headers = [
            'Date Acquired',
            'Date Sold',
            'Type',
            'Asset',
            'Quantity',
            'Cost Basis',
            'Proceeds',
            'Gain/Loss',
            'Holding Period',
            'Term Type'
        ];

        const rows = this.transactions.map(tx => {
            const gain = tx.proceeds - tx.costBasis;
            return [
                new Date(tx.acquiredDate).toISOString(),
                new Date(tx.soldDate).toISOString(),
                tx.type,
                tx.asset,
                tx.quantity.toFixed(6),
                tx.costBasis.toFixed(2),
                tx.proceeds.toFixed(2),
                gain.toFixed(2),
                this.calculateHoldingPeriod(tx.acquiredDate, tx.soldDate),
                this.isLongTerm(tx.acquiredDate, tx.soldDate) ? 'Long Term' : 'Short Term'
            ];
        });

        const csvContent = [
            headers.join(','),
            ...rows.map(row => row.join(','))
        ].join('\n');

        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.setAttribute('hidden', '');
        a.setAttribute('href', url);
        a.setAttribute('download', `crypto_tax_report_${new Date().getFullYear()}.csv`);
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    async generateForm8949() {
        try {
            const response = await fetch('/api/dashboard/generate-8949', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    year: new Date().getFullYear()
                })
            });

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.setAttribute('hidden', '');
            a.setAttribute('href', url);
            a.setAttribute('download', `form_8949_${new Date().getFullYear()}.pdf`);
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        } catch (error) {
            console.error('Failed to generate Form 8949:', error);
            alert('Failed to generate Form 8949. Please try again later.');
        }
    }
}

// Initialize tax manager when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.taxManager = new TaxManager();
}); 