const { test } = require('node:test');
const assert = require('node:assert/strict');
const h = require('../../assets/js/publication_helpers.js');

test('normalizes punctuation, accents, and AI terminology consistently', () => {
    for (const query of ['Human-AI', 'human–AI', 'human AI', 'human artificial intelligence']) {
        assert.equal(h.normalize(query), 'human ai');
    }
    assert.equal(h.normalize('García’s LLMs'), 'garcias llm');
});
test('short search terms match words, not the inside of unrelated words', () => {
    assert.equal(h.searchScore(h.searchIndex(['Human training with constraints'], 'Training'), 'human ai'), -1);
    assert.ok(h.searchScore(h.searchIndex(['Human–AI teaming'], 'Human–AI teaming'), 'human ai') > 0);
});
test('title matches rank ahead of abstract matches', () => {
    const title = h.searchIndex(['Generative design'], 'Generative design');
    const abstract = h.searchIndex(['A benchmark', 'Generative design methods'], 'A benchmark');
    assert.ok(h.searchScore(title, 'generative design') > h.searchScore(abstract, 'generative design'));
    assert.equal(h.searchScore(title, 'unrelated'), -1);
    assert.equal(h.searchScore(title, '  '), 0);
});
test('unsafe resource URLs and embedded credentials are rejected', () => {
    for (const url of ['javascript:alert(1)', 'data:text/html,foo', '//example.com', 'https://user:pass@example.com']) {
        assert.equal(h.safeURL(url), null);
    }
    assert.equal(h.safeURL('https://example.com/paper'), 'https://example.com/paper');
});
test('links distinguish DOI and preprint sources without inventing free access', () => {
    assert.deepEqual(h.resources({ pub_url: 'https://arxiv.org/abs/2601.12345' }), [
        { label: 'Read preprint', url: 'https://arxiv.org/abs/2601.12345' },
    ]);
    assert.deepEqual(h.resources({ doi: '10.1234/example', pub_url: 'https://doi.org/10.1234/example' }), [
        { label: 'DOI', url: 'https://doi.org/10.1234/example' },
    ]);
    assert.equal(h.resources({ pub_url: 'https://publisher.example/paper' })[0].label, 'Read paper');
    assert.deepEqual(h.resources({ pub_url: 'javascript:alert(1)' }), []);
    assert.equal(h.doiURL('not a doi'), null);
});
test('citation uses complete bibliographic venue and source link', () => {
    const text = h.citation({ pub_year: 2026, pub_url: 'https://doi.org/10.1234/example', bib_dict: {
        title: 'A study', author: 'A One and B Two', conference: 'Full conference name', citation: 'Full conference …, 2026',
    } });
    assert.equal(text, 'A One and B Two (2026). A study. Full conference name. https://doi.org/10.1234/example');
});
test('extracts only explicit DOIs in known publisher URL formats', () => {
    assert.equal(h.publisherDOI('https://dl.acm.org/doi/abs/10.1145/3600100.3626340'), 'https://doi.org/10.1145/3600100.3626340');
    assert.equal(h.publisherDOI('https://asmedigitalcollection.asme.org/computingengineering/article/doi/10.1115/1.4062852/1164198'), 'https://doi.org/10.1115/1.4062852');
    assert.equal(h.publisherDOI('https://www.emerald.com/insight/content/doi/10.1108/TPM-03-2019-0024/full/html'), 'https://doi.org/10.1108/TPM-03-2019-0024');
    assert.equal(h.publisherDOI('https://example.com/doi/10.1234/unverified'), null);
    assert.equal(h.publisherDOI('https://dl.acm.org/doi/%FF'), null);
});
