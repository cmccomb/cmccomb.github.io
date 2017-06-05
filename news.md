---
layout: page
title: News
permalink: /news/
published: true
order: 2
---

<div class="posts">
  {% for post in site.news reversed %}
  <div class="post">
    <h2 class="post-title">
      <a href="{{ post.url }}">
        {{ post.title }}
      </a>
    </h2>

    <span class="post-date">{{ post.date | date_to_string }}</span>

    {{ post.content }}
  </div>
  {% endfor %}
</div>
