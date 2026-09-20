/* Shared, deterministic publication formatting and search; no HTML from metadata. */
((root) => {
    "use strict";

    function normalize(value) {
        return String(value || "").normalize("NFKD")
            .replace(/\p{M}/gu, "").toLowerCase()
            .replace(/[’']/g, "")
            .replace(/[^\p{L}\p{N}]+/gu, " ")
            .replace(/\bartificial intelligence\b/g, "ai")
            .replace(/\bllms\b/g, "llm")
            .trim().replace(/\s+/g, " ");
    }

    function displayClusterLabel(label) {
        const value = String(label || "").trim();
        return value.toLowerCase() === "face to face" ? "design teams" : value;
    }

    function termsMatch(words, term) {
        return words.some(word => word === term || (term.length >= 4 && word.startsWith(term)));
    }

    function searchIndex(fields, title) {
        const text = normalize(fields.join(" "));
        const titleText = normalize(title);
        return { words: text.split(" "), title: titleText, titleWords: titleText.split(" ") };
    }

    function searchScore(index, query) {
        const phrase = normalize(query);
        const terms = phrase.split(" ").filter(Boolean);
        if (!terms.every(term => termsMatch(index.words, term))) return -1;
        if (!terms.length) return 0;
        return (index.title === phrase ? 1000 : 0)
            + (` ${index.title} `.includes(` ${phrase} `) ? 100 : 0)
            + terms.filter(term => termsMatch(index.titleWords, term)).length * 10;
    }

    function safeURL(value) {
        if (typeof value !== "string" || !/^https?:\/\//i.test(value)) return null;
        try {
            const url = new URL(value);
            return !url.username && !url.password ? url.href : null;
        } catch {
            return null;
        }
    }

    function doiURL(value) {
        const doi = String(value || "").trim().replace(/^(?:https?:\/\/(?:dx\.)?doi\.org\/|doi:\s*)/i, "");
        return /^10\.\d{4,9}\/\S+$/i.test(doi) ? `https://doi.org/${doi}` : null;
    }

    function publisherDOI(value) {
        const safe = safeURL(value);
        if (!safe) return null;
        const url = new URL(safe);
        let path;
        try { path = decodeURIComponent(url.pathname); } catch { return null; }
        if (/^(?:dx\.)?doi\.org$/.test(url.hostname)) return doiURL(path.slice(1));
        // Only known publisher URL formats; never infer a DOI from a paper title.
        if (url.hostname === "asmedigitalcollection.asme.org") {
            return doiURL(path.match(/\/doi\/(10\.1115\/[^/]+)/)?.[1]);
        }
        if (url.hostname === "www.emerald.com") {
            return doiURL(path.match(/\/doi\/(10\.1108\/[^/]+)\/full\/html$/)?.[1]);
        }
        if (["onlinelibrary.wiley.com", "www.tandfonline.com", "dl.acm.org",
            "www.liebertpub.com", "ascelibrary.org", "essopenarchive.org"].includes(url.hostname)) {
            return doiURL(path.match(/\/doi\/(?:abs\/|full\/|pdf\/|epdf\/)?(10\..+)$/)?.[1]);
        }
        return null;
    }

    function resources(record) {
        const bib = record.bib_dict || {};
        const links = [];
        const add = (label, value) => {
            const url = safeURL(value);
            if (url && !links.some(link => link.url === url)) links.push({ label, url });
        };
        const paperURL = safeURL(record.pub_url || bib.url);
        add("DOI", doiURL(record.doi || bib.doi) || publisherDOI(paperURL));
        if (paperURL) {
            const host = new URL(paperURL).hostname;
            const label = /^(?:dx\.)?doi\.org$/.test(host) ? "DOI"
                : /^(?:www\.)?(?:arxiv\.org|biorxiv\.org|medrxiv\.org)$/.test(host)
                    ? "Read preprint" : "Read paper";
            add(label, paperURL);
        }
        add("Open-access version", record.eprint_url);
        return links;
    }

    function authors(value) {
        const parts = String(value || "").split(" and ").filter(Boolean);
        if (parts.length > 2) return `${parts.slice(0, -1).join(", ")}, and ${parts.at(-1)}`;
        return parts.join(" and ");
    }

    function venue(record) {
        const bib = record.bib_dict || {};
        const name = bib.journal || bib.conference;
        if (!name) return String(bib.citation || "");
        const volume = bib.volume ? `${bib.volume}${bib.number ? `(${bib.number})` : ""}` : "";
        return [name, volume, bib.pages ? `pp. ${bib.pages}` : ""].filter(Boolean).join(", ");
    }

    function citation(record) {
        const bib = record.bib_dict || {};
        const link = resources(record)[0]?.url;
        return [
            `${authors(bib.author)} (${record.pub_year}).`.trim(),
            `${String(bib.title || "").replace(/[.!?]$/, "")}.`,
            venue(record) ? `${venue(record).replace(/\.$/, "")}.` : "",
            link,
        ].filter(Boolean).join(" ");
    }

    const helpers = { normalize, displayClusterLabel, searchIndex, searchScore, safeURL, doiURL, publisherDOI, resources, authors, venue, citation };
    if (typeof module !== "undefined" && module.exports) module.exports = helpers;
    else root.PublicationHelpers = helpers;
})(typeof window !== "undefined" ? window : globalThis);
