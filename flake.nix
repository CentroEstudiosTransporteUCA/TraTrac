{
  description = "TraTrac dev shell: uv-managed Python 3.12, with the shared libs manylinux wheels need on NixOS";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs:
        let
          # numpy/opencv-python/torch etc. are manylinux wheels: they expect a
          # standard FHS dynamic-linker search path, which NixOS doesn't provide.
          # Put the shared libs they dlopen at runtime (libstdc++, libGL, the X11/
          # glib stack opencv-python's GUI bindings pull in, ...) on the loader path.
          libPath = pkgs.lib.makeLibraryPath (with pkgs; [
            stdenv.cc.cc.lib
            zlib
            libGL
            glib.out
            libSM
            libICE
            libxcb
            libxext
            libxrender
            libx11
            libxau
            libxdmcp
            fontconfig.lib
            freetype
            libxkbcommon
            dbus
          ]);
        in
        {
          default = pkgs.mkShell {
            packages = [ pkgs.uv pkgs.python312 ];

            LD_LIBRARY_PATH = libPath;

            # uv's own standalone Python downloads are dynamically linked for a
            # generic glibc host and refuse to run on NixOS (see
            # nix.dev/permalink/stub-ld) - always use the Nix-provided
            # interpreter instead of letting uv fetch its own.
            UV_PYTHON = "${pkgs.python312}/bin/python3.12";
            UV_PYTHON_DOWNLOADS = "never";
          };
        });
    };
}
