---
layout: page
title: Software
permalink: /cv/
order: 3
---

<table style="border:none ;">
{% for repository in site.github.public_repositories %}
  <tr>
    <td style="vertical-align: middle; border: none;">
      <h3>
        <a href="{{ repository.homepage }}">
          {{ repository.name }}
        </a>
      </h3>
    </td>
    <td style="vertical-align: top; border: none; white-space: nowrap;">
        Language: <code>{{ repository.language }}</code><br/>
        Status: <a href="https://travis-ci.org/HSDL/{{ repository.name }}" style="margin: 0; padding: 0;">
        <img src="https://travis-ci.org/HSDL/{{ repository.name }}.svg?branch=master" style="margin: 0; padding: 0;  margin-bottom: -6px"></a><br/>
        Last updated on {{ repository.pushed_at | date_to_string }}
    </td>
    <td style="vertical-align: top; border: none;">
      <span style="font-style: italic">{{repository.description}}</span>
    </td>
  </tr>
{% endfor %}
</table>
