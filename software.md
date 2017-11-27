---
layout: page
title: Software
permalink: /software/
order: 4
---

<table>
  <tr>
    <th>Name</th>
    <th>Info</th>
    <th>Description</th>
  </tr>
{% for repository in site.github.public_repositories %}
  <tr>
    <td style="vertical-align: top;">
        <a href="{{ repository.repository_url }}">
          {{ repository.name }}
        </a>
    </td>
    <td style="vertical-align: top;  white-space: nowrap;">
        Language: <code>{{ repository.language }}</code><br/>
        License: <code>{{ repository }}</code><br/>
        Last updated on {{ repository.pushed_at | date_to_string }}
    </td>
    <td style="vertical-align: top;">
      <span style="font-style: italic">{{repository.description}}</span>
    </td>
  </tr>
{% endfor %}
</table>
