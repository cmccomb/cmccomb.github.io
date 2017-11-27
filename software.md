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
    <td style="vertical-align: middle; border: none;">
      <h3>
        <a href="{{ repository.homepage }}">
          {{ repository.name }}
        </a>
      </h3>
    </td>
    <td style="vertical-align: top; border: none; white-space: nowrap;">
        Language: <code>{{ repository.language }}</code><br/>
        Last updated on {{ repository.pushed_at | date_to_string }}
    </td>
    <td style="vertical-align: top; border: none;">
      <span style="font-style: italic">{{repository.description}}</span>
    </td>
  </tr>
{% endfor %}
</table>
