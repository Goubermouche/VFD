workspace "FluidEngine"
    architecture "x64"

    configurations
    {
        "Debug",
        "Release"
    }

    startproject "FluidEngine"

outputdir = "{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"
-- Include directories relative to root folder (solution directory).
IncludeDir = {}
IncludeDir["GLFW"] = "FluidEngine/Vendor/GLFW/include"
IncludeDir["Glad"] = "FluidEngine/Vendor/Glad/include"
IncludeDir["ImGui"]= "FluidEngine/Vendor/imgui"
IncludeDir["glm"]  = "FluidEngine/Vendor/glm"
IncludeDir["entt"] = "FluidEngine/Vendor/entt/include"

include "FluidEngine/Vendor/GLFW"
include "FluidEngine/Vendor/Glad"
include "FluidEngine/Vendor/imgui"

project "FluidEngine"
    location "FluidEngine"
    kind "ConsoleApp"
    language "C++"

    targetdir "bin/%{cfg.buildcfg}"

    pchheader "pch.h"
    pchsource "FluidEngine/src/pch.cpp"

    files
    {
        "%{prj.name}/src/**.h",
        "%{prj.name}/src/**.cpp",
        "%{prj.name}/src/**.cu",
        "%{prj.name}/src/**.cuh",
        "%{prj.name}/src/**.txt",
        "%{prj.name}/src/**.shader",

        -- "%{prj.name}/Vendor/stb/**.h",
        -- "%{prj.name}/Vendor/stb/**.cpp",
        "%{prj.name}/Vendor/OBJ-loader/**.h",
        "%{prj.name}/Vendor/glm/**.hpp",
        "%{prj.name}/Vendor/glm/**.inl",
    }

    includedirs
    {
        "%{prj.name}/$(ProjectDir)src",
        "%{IncludeDir.GLFW}",
        "%{IncludeDir.Glad}",
        "%{IncludeDir.ImGui}",
        "%{IncludeDir.glm}",
        "%{IncludeDir.entt}",
        -- "%{prj.name}/$(ProjectDir)Vendor/stb/include",
    }

    links
    {
        "GLFW",
        "Glad",
        "ImGui",
        "opengl32.lib"
    }

    filter "system:windows"
        cppdialect "C++20"
        staticruntime "On"
        systemversion "latest"

        defines
        {
            "GLFW_INCLUDE_NONE"
        }

    filter "configurations:Debug"
        defines { "DEBUG" }
        symbols "On"
        staticruntime "off"
  
     filter "configurations:Release"
        defines { "NDEBUG" }
        optimize "On"
        staticruntime "off"
