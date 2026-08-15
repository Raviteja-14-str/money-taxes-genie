# Money Genie knowledge-source policy

Updated: 2026-08-15

Money Genie is a financial-literacy assistant for India. The knowledge files are curated summaries, not a dump of the internet. Each new topic should prefer:

1. Primary Indian sources: Income Tax Department, RBI, SEBI, IRDAI, EPFO, PFRDA, NPCI, DICGC, GSTN/GST Council, and India Post.
2. Official consumer-education material from the relevant regulator.
3. Reputable open-source projects only for engineering patterns, test ideas, and taxonomy. Do not copy their user data, proprietary text, credentials, or code unless its license allows reuse and the project explicitly needs it.

## Source register

- Income tax: [Income Tax Department salary and regime guidance](https://www.incometax.gov.in/iec/foportal/help/individual/return-applicable-1), [AIS FAQ](https://www.incometax.gov.in/iec/foportal/ais-faq), and [ITR FAQ](https://www.incometax.gov.in/iec/foportal/node/11724).
- Banking and consumer protection: [RBI FAME](https://www.rbi.org.in/commonperson/English/Scripts/PressReleases.aspx?Id=2123), [RBI customer awareness](https://systemhealth.rbi.org.in/rbikehtahai.rbi.org.in/RBI%20Kehta%20Hai%20!_files/rkh.html), and [DICGC deposit insurance](https://www.dicgc.org.in/guide-to-deposit-insurance).
- Securities and mutual funds: [SEBI Investor education](https://investor.sebi.gov.in/iematerial.html) and [Understanding Mutual Funds](https://investor.sebi.gov.in/understanding_mf.html).
- Insurance: [IRDAI health guidance](https://irdai.gov.in/health-dept) and [IRDAI consumer affairs](https://irdai.gov.in/consumer-affairs-booklet1).
- Retirement: [EPFO overview](https://www.epfindia.gov.in/site_en/AboutEPFO.php), [PFRDA NPS](https://www.pfrda.org.in/en/schemes/national-pension-system/about-nps), and [India Post savings schemes](https://www.indiapost.gov.in/banking-services/savings).
- Digital payments: [NPCI UPI safety campaign](https://www.npci.org.in/PDF/npci/press-releases/2024/NPCI-Press-release-NPCI-Unveils-UPI-Safety-Awareness-Campaign-to-Champion-Safe-Digital-Payment-Practices).

## Update rules

- Add `SOURCE NOTES` and a `checked YYYY-MM-DD` date to every new knowledge file.
- Keep changing numbers, rates, thresholds, due dates, and portal instructions in clearly dated documents.
- When a rule changes, replace or supersede the old document rather than leaving two contradictory copies active.
- Keep product recommendations, live prices, and user-specific calculations out of static educational files.
- Re-run the evaluation questions after every tax-year or major regulator update.

## What “train the model” means here

The local project uses retrieval-augmented generation (RAG): it retrieves relevant files and places them in the model context. This is the right first mechanism for changing tax and finance information because the source files can be reviewed and updated without retraining model weights. Fine-tuning can be considered later for tone, multilingual phrasing, or classification, but it should not be used as the source of current tax rates or legal rules.
