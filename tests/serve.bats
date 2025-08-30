#!/usr/bin/env bats

setup() {
	STUB_DIR="$BATS_TEST_DIRNAME/stub"
	mkdir -p "$STUB_DIR"
	export PATH="$STUB_DIR:$PATH"
}

teardown() {
	rm -rf "$STUB_DIR"
}

@test "forwards JEKYLL_ENV and arguments" {
	cat <<'STUB' >"$STUB_DIR/bundle"
#!/usr/bin/env bash
echo "args: $*"
echo "env: $JEKYLL_ENV"
STUB
	chmod +x "$STUB_DIR/bundle"

	export JEKYLL_ENV=production
	run "$BATS_TEST_DIRNAME/../_scripts/serve.sh"
	[ "$status" -eq 0 ]
	[[ "$output" == *"args: exec jekyll serve"* ]]
	[[ "$output" == *"env: production"* ]]
}

@test "propagates exit status" {
	cat <<'STUB' >"$STUB_DIR/bundle"
#!/usr/bin/env bash
exit 42
STUB
	chmod +x "$STUB_DIR/bundle"

	run "$BATS_TEST_DIRNAME/../_scripts/serve.sh"
	[ "$status" -eq 42 ]
}
